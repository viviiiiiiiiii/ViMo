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
                        repetition_penalty=1.15,
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

template_universale = """You are a highly analytical Multimodal QA agent. You CAN see the images attached to this message directly.
You are STRICTLY FORBIDDEN from starting lines with anything other than 'Thought:', 'Action:', 'Action Input:', or 'Final Answer:'.

TOOLS AVAILABLE:
{tools}

CRITICAL RULES:
1. VISUALIZE: The user has attached an image. Look at it to understand the subject.
2. START WITH TOOL: Even though you can see the image, you MUST use 'tool_ricerca_visiva' passing the image filename to fetch the official Wikipedia documents.
3. ANTI-LOOP PROTOCOL: You are STRICTLY FORBIDDEN from reading the exact same section of the same document twice.
4. MANDATORY THOUGHT CHECKLIST: Your 'Thought:' MUST be a single paragraph with 4 numbered steps. Do not use brackets like [ ].

MANDATORY FORMAT:
Thought: 1) Task: [What to do] 2) Visuals: [What you see directly in the image and what the tool found] 3) Evaluation: [Compare findings] 4) Next: [Next tool to use and WHY]
Action: [{tool_names}]
Action Input: [The exact tool input]
Observation: [Result from the tool]

... (Repeat until you have the answer)

Thought: 1) Task: Answer user. 2) Visuals: Analyzed. 3) Evaluation: All data gathered. 4) Next: Provide Final Answer.
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


    # Gli passiamo il path tecnico e poi la domanda "umana" senza confonderlo col nome file
    input_semplice = f"[IMG]{percorso_immagine}[/IMG]\nGuarda l'immagine allegata. Usa 'tool_ricerca_visiva' per trovare l'opera e il soggetto nei documenti ufficiali. Poi leggi i documenti Wikipedia trovati per scoprire l'autore dell'opera e indicami le sue invenzioni più famose."

    print(f"\n🧠 Avvio indagine di Qwen. Occhi puntati su: {percorso_immagine}...")
    esecutore.invoke({"input": input_semplice})