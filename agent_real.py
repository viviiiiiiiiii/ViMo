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

template_universale = """You are an elite, highly critical investigative research assistant. Your goal is to provide accurate answers through methodical research.

TOOLS AVAILABLE:
{tools}

RULES OF ENGAGEMENT:
1. TWO-STEP PROCESS: First, use a search tool (visual or text) to find relevant documents. You will receive a summary of MULTIPLE documents.
2. VISUAL GROUNDING (CRITICAL): Whenever you have an input image, explicitly describe what you see in it within your Thought. 
3. EVALUATE ALL BEFORE ACTING: Read the titles and section lists of ALL retrieved documents. Compare them with your visual analysis of the image. Choose the document that logically matches both the image content AND the specific information requested.
4. READING: Use 'tool_leggi_sezione' to read a specific section. Format: URL_DOC | NUMERO_SEZIONE | SI
5. EXHAUSTIVE BACKTRACKING: If a section does not contain the answer:
   - Level 1: Use 'tool_leggi_sezione' on a DIFFERENT section of the SAME document.
   - Level 2: Switch to a DIFFERENT document from your previous search results.
6. TRACK YOUR PROGRESS: NEVER read the same section of the same document twice.
7. SEPARATOR: You MUST use the pipe symbol '|' for tool_leggi_sezione.
8. STRICT FORMAT: NEVER output conversational text. EVERY single line MUST begin with 'Thought:', 'Action:', 'Action Input:', or 'Final Answer:'.

========================================
EXAMPLE OF A PERFECT EXECUTION (FORMAT ONLY):
Question: Guarda l'immagine. Usa la ricerca visiva per capire che monumento è. Poi leggi i documenti per scoprire in che anno è stato inaugurato.
Thought: 1. Nell'immagine vedo una grande torre di metallo. 2. Devo fare una ricerca visiva per identificarne il nome esatto.
Action: tool_ricerca_visiva
Action Input: foto_monumento.jpg
Observation: [URL_DOC: http://wiki/Paris] (Sezioni: 0, 1), [URL_DOC: http://wiki/Eiffel_Tower] (Sezioni: 0, 1, 2)
Thought: 1. I risultati mostrano "Paris" e "Eiffel Tower". 2. Dato che cerco informazioni specifiche sul monumento e sulla sua inaugurazione, il documento "Eiffel Tower" è il più pertinente. 3. Scelgo di leggere la sezione 1 che parla della storia.
Action: tool_leggi_sezione
Action Input: http://wiki/Eiffel_Tower | 1 | SI
Observation: La torre è stata inaugurata il 31 marzo 1889 in occasione dell'Esposizione Universale.
Thought: 1. Ho trovato la data di inaugurazione nel testo: 1889. 2. Ho tutte le informazioni necessarie per rispondere all'utente.
Final Answer: Il monumento nell'immagine è la Torre Eiffel ed è stato inaugurato nel 1889.
========================================

MANDATORY FORMAT:
Thought: [Your step-by-step reasoning based on the formatting logic of the example above]
Action: [{tool_names}]
Action Input: [The specific query]
Observation: [Result from the tool]

... (Repeat if needed)

Thought: I have gathered all necessary information.
Final Answer: [Your comprehensive answer]

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

    input_semplice = f"Guarda l'immagine '{percorso_immagine}'. Cerca chi l'ha dipinta usando la ricerca visiva. Poi leggi la sezione che parla delle sue invenzioni usando tool_leggi_sezione e dimmi cosa trovi."
    
    print(f"\n🧠 Avvio indagine di Qwen. Occhi puntati su: {percorso_immagine}...")
    esecutore.invoke({"input": input_semplice})