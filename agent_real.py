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

template_universale = """You are a highly analytical, robotic Vision-QA agent. You DO NOT possess conversational abilities. 
You are STRICTLY FORBIDDEN from wrapping your thoughts in brackets like [ ] or using conversational preambles. 
EVERY line you generate MUST begin with 'Thought:', 'Action:', 'Action Input:', or 'Final Answer:'.

TOOLS AVAILABLE:
{tools}

CRITICAL RULES FOR REASONING AND EVALUATION:
1. START VISUALLY: Always use 'tool_ricerca_visiva' first to identify the context of the image.
2. NO GUESSWORK OR LOOPING: You are FORBIDDEN from reading sections sequentially (e.g., section 1, then 2, then 3). 
3. MANDATORY GLOBAL EVALUATION: When a tool returns multiple documents, you MUST read all titles and their respective section names. You MUST evaluate which specific document AND which specific section matches the user's exact question before using the reading tool.
4. READING TOOL: Use 'tool_leggi_sezione' passing exactly: URL_DOC | NUMERO_SEZIONE | SI

MANDATORY FORMAT:
Thought: 1) What the user is asking. 2) Evaluation of ALL retrieved document titles and sections. 3) Logical deduction of the SINGLE best document and section to read.
Action: [{tool_names}]
Action Input: [The exact tool input]
Observation: [Result from the tool]

... (Repeat ONLY if the chosen section was definitively wrong)

Thought: I have the information needed to answer the user's question.
Final Answer: [Your precise and direct answer]

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