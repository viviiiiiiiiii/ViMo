import base64
import torch
import re
import os
from typing import Optional, List

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

import tools_real 
from load_config import load_config
from Qwen_retrieval import generate_answer
from eval_utils import (
    build_common_record, elapsed, now_seconds, 
    parse_retrieved_urls, retrieval_metrics, token_estimate,
)

# Configurazione
config_dict = load_config()
class Args: pass
args = Args()
for key, value in config_dict.items(): setattr(args, key, str(value))
args.top_k = 3

tools_real.start_motors(args)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


class QwenServerLLM(LLM):
    current_image_path: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom-multimodal"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
            
            user_content = []
            
            # 1. TRUCCO DEL TAG: Intercettiamo l'immagine nascosta nel prompt di LangChain
            match = re.search(r'\[IMG\](.*?)\[/IMG\]', prompt)
            if match:
                image_path = match.group(1).strip()
                if os.path.exists(image_path):
                    user_content.append({"type": "image", "image": image_path})
                # Rimuoviamo il tag dal testo per non confondere Qwen
                prompt = re.sub(r'\[IMG\].*?\[/IMG\]\n?', '', prompt)
            # Fallback di sicurezza se usi il vecchio metodo
            elif self.current_image_path and os.path.exists(self.current_image_path):
                user_content.append({"type": "image", "image": self.current_image_path})
            
            # 2. Immagini wiki scaricate al volo dal tool
            wiki_images = re.findall(r'\[IMG_WIKI:\s*(.*?)\]', prompt)
            for img_path in set(wiki_images): 
                if os.path.exists(img_path):
                    user_content.append({"type": "image", "image": img_path})
                    print(f"👁️ Qwen sta guardando un'immagine extra da Wikipedia: {img_path}")
            
            user_content.append({"type": "text", "text": prompt})

            messages = [
                {"role": "system", "content": "You are a rigid, robotic backend system. You MUST communicate ONLY using the requested ReAct format (Thought, Action, Action Input). NO conversational filler."},
                {"role": "user", "content": user_content}
            ]
            
            if tools_real.qwen_model is None or tools_real.qwen_processor is None:
                raise ValueError("Errore: I motori del server non sono stati accesi!")

            risposta = generate_answer(
                        tools_real.qwen_model, 
                        tools_real.qwen_processor, 
                        messages,
                        #repetition_penalty=1.15,
                        max_new_tokens=512 # Aumentato per evitare tagli a metà frase
                    )

            manual_stops = (stop or []) + ["Observation:", "Observation", "\nObservation:"]
            for stop_word in manual_stops:
                if stop_word in risposta:
                    risposta = risposta.split(stop_word)[0]

            return risposta.strip()


# ==========================================
# SETUP AGENTE - NUOVO PROMPT A DUE FASI
# ==========================================

template_universale = """You are a visual AI agent. You CAN see the attached image.
Answer the following questions as best you can. You have access to the following tools:

{tools}

STRICT RULES:
1. You MUST use 'tool_ricerca_visiva' FIRST to understand what the image is.
2. Your 'Thought' MUST be a single line. Do NOT use newlines.
3. Do NOT hallucinate URLs. Only use URLs exactly as returned by your tools.

Use the following exact format:

Question: the input question you must answer
Thought: you should always think about what to do next on a SINGLE line
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    template=template_universale,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)

vero_qwen = QwenServerLLM()

agente = create_react_agent(vero_qwen, tools_real.miei_tools_reali, prompt)
esecutore = AgentExecutor(
    agent=agente, 
    tools=tools_real.miei_tools_reali, 
    verbose=True,
    handle_parsing_errors="Check your output format! Remember to use Action: and Action Input:.",
    max_iterations=6, 
    early_stopping_method='force',
    return_intermediate_steps=True
)

def _summarize_intermediate_steps(intermediate_steps):
    # La tua logica di estrazione metadati (fondamentale per la valutazione)
    tool_calls =[]
    all_observations = []
    for step in intermediate_steps or[]:
        action, observation = step
        tool_calls.append({"tool": getattr(action, "tool", None), "tool_input": str(getattr(action, "tool_input", None))})
        all_observations.append(str(observation))
    
    retrieved_urls =[]
    for obs in all_observations: retrieved_urls.extend(parse_retrieved_urls(obs))
    
    return {
        "tool_calls": tool_calls,
        "retrieved_urls": list(set(retrieved_urls)),
        "num_tool_calls": len(tool_calls),
        "observations_tokens_est": token_estimate("\n".join(all_observations)),
    }

def run_agentic_rag(image_path, question, question_id=None, ground_truth="", question_type="unknown", expected_sources=None):
    vero_qwen.current_image_path = image_path
    start = now_seconds()
    result = {}
    try:
        result = esecutore.invoke({"input": question})
        answer = result.get("output", "")
        error = None
    except Exception as e:
        answer = ""
        error = str(e)
        
    step_info = _summarize_intermediate_steps(result.get("intermediate_steps",[]))
    ret_metrics = retrieval_metrics(step_info.get("retrieved_urls",[]), expected_sources or[], k=args.top_k)
    
    return build_common_record(
        question_id=question_id, model_name="agentic_rag", image_path=image_path,
        question=question, ground_truth=ground_truth, answer=answer, 
        question_type=question_type, latency_seconds=elapsed(start), error=error,
        extra={"num_steps": step_info["num_tool_calls"], **step_info, **ret_metrics}
    )
    

if __name__ == "__main__":
    print("🚀 Inizializzazione sistema sul server...")    
    
    percorso_immagine = "esempio3.jpg"
    vero_qwen.current_image_path = percorso_immagine

    # 2. DOMANDA TECNICA: Materiali e specifiche di peso
    domanda = "Identify this structure. What specific type of iron was used in its construction and what is the estimated weight of the metal framework alone?" 

    input_semplice = f"[IMG]{percorso_immagine}[/IMG]\nLook at the attached image (filename: {percorso_immagine}). Then, read the correct document to find the technical answer: {domanda}"

    print(f"\n🧠 Avvio indagine TECNICA di Qwen. Target: {percorso_immagine}...")
    
    risultato_finale = esecutore.invoke({"input": input_semplice})
    
    print("\n" + "🔥"*25)
    print("🎯 RISPOSTA FINALE TECNICA:")
    if isinstance(risultato_finale, dict) and "output" in risultato_finale:
        print(risultato_finale["output"])
    else:
        print(risultato_finale)
    print("🔥"*25 + "\n")