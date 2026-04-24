

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
        # Formattazione del prompt
        messages = [{"role": "user", "content": prompt}]
        
        if tools_real.qwen_model is None or tools_real.qwen_processor is None:
            raise ValueError("Errore: I motori del server non sono stati accesi!")

        # 1. Generiamo la risposta completa da Qwen
        risposta = generate_answer(
            tools_real.qwen_model, 
            tools_real.qwen_processor, 
            messages
            # Rimuoviamo stop=stop da qui perché model.generate spesso lo ignora
        )

        # 2. 📍 LA MAGIA: Applichiamo il "Freno a mano" di LangChain manualmente
        # LangChain passerà stop=["\nObservation:", "Observation:"]
        if stop is not None:
            for stop_word in stop:
                if stop_word in risposta:
                    # Se Qwen ha provato a scrivere "Observation:", tagliamo la frase lì!
                    risposta = risposta.split(stop_word)[0]

        # Restituiamo la stringa pulita e tagliata
        return risposta.strip()

# ==========================================
# SETUP AGENTE (Globali)
# ==========================================
# 📍 Modificato il template per essere 100% compatibile con create_react_agent
# In agent_real.py

# Modifica il template in agent_real.py
# ==========================================
# SETUP AGENTE (Globali) - VERSIONE GROUNDED
# ==========================================

template_universale = """You are a strictly grounded assistant. Answer the following questions based ONLY on information retrieved from tools.

You have access to the following tools:
{tools}

STRICT RULES:
1. Use ONLY the information provided in the 'Observation' sections.
2. If the tools do not provide information about a specific request (e.g. inventions), do NOT invent them. State: "Information not found in the database."
3. Never use your internal knowledge to supplement the database. 
4. If you identify a person but the tool doesn't mention their inventions, you MUST try 'tool_ricerca_testuale' before giving up.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

# Il resto del codice (PromptTemplate, agente, esecutore) rimane uguale
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