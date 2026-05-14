import base64
import torch
import re
import os
from PIL import Image
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
            
            # 1. Immagine principale dell'utente
            if self.current_image_path and os.path.exists(self.current_image_path):
                user_content.append({"type": "image", "image": self.current_image_path})
            
            # 2. Immagini wiki scaricate al volo dal tool
            wiki_images = re.findall(r'\[IMG_WIKI: (.*?)\]', prompt)
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
                        repetition_penalty=1.15,
                        max_new_tokens=256       
                    )

            manual_stops = (stop or []) + ["Observation:", "Observation", "\nObservation:"]
            for stop_word in manual_stops:
                if stop_word in risposta:
                    risposta = risposta.split(stop_word)[0]

            return risposta.strip()


# ==========================================
# SETUP AGENTE - NUOVO PROMPT A DUE FASI
# ==========================================

template_universale = """You are a rigid, robotic Vision-QA agent executing the Encyclopedic-VQA benchmark. 
You DO NOT have conversational capabilities. You are FORBIDDEN from starting your response with conversational text, greetings, or direct descriptions.
YOUR VERY FIRST WORD MUST ALWAYS BE "Thought:".

TOOLS AVAILABLE:
{tools}

STRICT WORKFLOW FOR SINGLE-HOP VQA:
1. VISUAL SEARCH: You must ALWAYS start by using 'tool_ricerca_visiva' to identify the entity in the input image.
2. EVALUATE: Compare the retrieved Wikipedia document titles with the image and the user's specific question.
3. READ: Use 'tool_leggi_sezione' to read the text of the most relevant document. You MUST use the pipe separator (Example: http://url.com/ | 1 | SI).
4. ANSWER: Extract the exact answer to the user's question from the text you just read and output the Final Answer.

MANDATORY FORMAT (No exceptions, no preambles):
Thought: [Describe the image internally, evaluate options, state your next tool]
Action: [{tool_names}]
Action Input: [The input for the tool]
Observation: [Result from the tool]
... (repeat if necessary)
Thought: I have found the precise answer in the text.
Final Answer: [Your concise and direct answer]

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
    early_stopping_method='force'
)

def run_agentic_rag(image_path, question):
    vero_qwen.current_image_path = image_path
    try:
        result = esecutore.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Errore Agente: {str(e)}"

if __name__ == "__main__":
    print("🚀 Inizializzazione sistema sul server...")
    
    config_dict = load_config()

    class CostruttoreArgs: pass
    args = CostruttoreArgs()
    for key, value in config_dict.items():
        setattr(args, key, str(value))
    args.top_k = 3
    
    tools_real.start_motors(args)
    
    percorso_immagine = "foto_buia.jpg"
    vero_qwen.current_image_path = percorso_immagine

    # Passa la domanda in modo secco, l'Agente sa già che deve usare la ricerca visiva
    domanda_vqa = "Who painted this masterpiece and what are some of his famous inventions?"
    
    input_semplice = f"Image: {percorso_immagine}\nQuestion: {domanda_vqa}"
    
    print(f"\n🧠 Avvio indagine di Qwen. Occhi puntati su: {percorso_immagine}...")
    esecutore.invoke({"input": input_semplice})