

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
# L'ADATTATORE QWEN (Corretto)
# ==========================================
class QwenServerLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # Implementazione pulita per evitare il TypeError
        messages = [{"role": "user", "content": prompt}]
        
        if tools_real.qwen_model is None or tools_real.qwen_processor is None:
            raise ValueError("Errore: I motori del server non sono stati accesi!")

        return generate_answer(
            tools_real.qwen_model, 
            tools_real.qwen_processor, 
            messages,
            stop=stop
        )

# ==========================================
# SETUP AGENTE (Globali)
# ==========================================
# 📍 Modificato il template per essere 100% compatibile con create_react_agent
# In agent_real.py

# Modifica il template in agent_real.py
template_istruzioni = """Rispondi alla domanda dell'utente nel miglior modo possibile. Hai accesso ai seguenti strumenti:

{tools}

Devi usare ESATTAMENTE questo formato rigido:

Question: la domanda a cui devi rispondere
Thought: pensa sempre a cosa devi fare passo dopo passo
Action: l'azione da eseguire, deve essere UNA SOLA tra [{tool_names}]
Action Input: l'input per l'azione (es. foto_buia.jpg)
Observation: il risultato dell'azione
... (questo ciclo Thought/Action/Action Input/Observation può ripetersi N volte)
Thought: Ora so la risposta finale
Final Answer: la risposta finale alla domanda originale

INIZIA!

Question: {input}
Thought: {agent_scratchpad}"""

# Assicurati che il PromptTemplate dichiari TUTTE le variabili necessarie
prompt = PromptTemplate(
    template=template_istruzioni,
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
    percorso_immagine = "foto_buia.jpg" 
    input_semplice = f"Ho un'immagine chiamata '{percorso_immagine}'. Per favore, usa il tool visivo per dirmi chi è l'autore del quadro."

    print("\n🤖 Agente in ascolto (Stabile)...")
    esecutore.invoke({"input": input_semplice})