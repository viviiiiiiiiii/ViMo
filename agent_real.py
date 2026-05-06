

import base64
import torch
import re
from PIL import Image
from typing import Optional, List

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 📍 Import dal core (sempre validi)
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

# 📍 Import per AgentExecutor e ReAct (Versione 2026 / Classic)
# Proviamo i due percorsi più probabili per la v1.2.15
try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

# Importiamo l'intero modulo per accedere alle variabili globali aggiornate
import tools_real 
from load_config import load_config
from Qwen_retrieval import generate_answer


# ==========================================
# FUNZIONI DI SUPPORTO
# ==========================================
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# ==========================================
# L'ADATTATORE QWEN (Corretto con Freno a Mano)
# ==========================================
class QwenServerLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
            messages = [{"role": "user", "content": prompt}]
            
            if tools_real.qwen_model is None or tools_real.qwen_processor is None:
                raise ValueError("Errore: I motori del server non sono stati accesi!")

            # 1. Chiamiamo la generazione SENZA stop_words (che causava l'errore)
            risposta = generate_answer(
                        tools_real.qwen_model, 
                        tools_real.qwen_processor, 
                        messages,
                        do_sample=False,         # 📍 DISATTIVA il campionamento (Greedy decoding)
                        repetition_penalty=1.5,  # 📍 Alza ancora per stroncare i "blissfully"
                        max_new_tokens=256       # Riduciamo per sicurezza
                    )
            # Se la risposta è troppo lunga e non contiene "Action:", è un loop
            if len(risposta) > 300 and "Action:" not in risposta:
                return "Thought: I am stuck in a loop. I need to rethink.\nAction: tool_ricerca_visiva\nAction Input: foto_buia.jpg"
            
            # 2. Gestiamo gli STOP WORDS manualmente qui (Freno a mano software)
            # Se Qwen prova a scrivere "Observation:" da solo, noi lo tagliamo fuori.
            manual_stops = (stop or []) + ["Observation:", "Observation", "\nObservation:"]
            
            for stop_word in manual_stops:
                if stop_word in risposta:
                    # Teniamo solo quello che c'è PRIMA della parola di stop
                    risposta = risposta.split(stop_word)[0]

            return risposta.strip()


# ==========================================
# SETUP AGENTE (Globali) - VERSIONE CORRETTA
# ==========================================

template_universale = """You are a precise research assistant. 
Use ONLY the following tools. 

Tools:
{tools}

Format to follow:
Thought: I need to use a tool.
Action: [Tool Name ONLY, e.g., tool_ricerca_visiva]
Action Input: [The input, e.g., foto_buia.jpg]
Observation: [Wait]

... (this can repeat)

Thought: I have the final answer.
Final Answer: [The summary]

Begin!
Question: {input}
Thought: {agent_scratchpad}"""
# Assicurati che PromptTemplate rimanga così:
prompt = PromptTemplate(
    template=template_universale,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)

vero_qwen = QwenServerLLM()

# Inizializziamo l'agente e l'esecutore
agente = create_react_agent(vero_qwen, tools_real.miei_tools_reali, prompt)
esecutore = AgentExecutor(
    agent=agente, 
    tools=tools_real.miei_tools_reali, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5, # Per evitare loop infiniti
    early_stopping_method='force'
)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🚀 Inizializzazione sistema sul server...")
    
    # 1. Caricamento configurazione
    config_dict = load_config()

    class CostruttoreArgs:
        pass
    
    args = CostruttoreArgs()
    for key, value in config_dict.items():
        setattr(args, key, str(value))
    args.top_k = 3
    
    # 2. ACCENSIONE MOTORI (Popola tools_real.qwen_model, ecc.)
    tools_real.start_motors(args)
    
    # 3. TEST AGENTE
# 3. TEST AGENTE SOLO TESTO
percorso_immagine = "foto_buia.jpg"
input_semplice = "Identifica il soggetto in 'foto_buia.jpg'. Una volta capito chi è, usa la ricerca testuale per dirmi quali sono le sue invenzioni citate nel database che NON siano quadri."
esecutore.invoke({"input": input_semplice})