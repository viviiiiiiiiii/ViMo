

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

template_istruzioni = """Sei un assistente esperto d'arte. Hai accesso ai seguenti strumenti:

{tools}

Per rispondere alla domanda, segui sempre questo formato:

Thought: Rifletti su cosa devi fare.
Action: Il nome dello strumento da usare (deve essere uno tra: {tool_names})
Action Input: L'input per lo strumento (ad esempio il nome del file immagine o una parola chiave)
Observation: Il risultato dell'azione (questo viene fornito dal sistema)

... (ripeti se necessario)

Thought: Ora conosco la risposta finale.
Final Answer: La risposta conclusiva alla domanda originale.

IMPORTANTE: Fermati SEMPRE dopo aver scritto 'Action Input:'. Non inventare mai l'Observation.

Domanda: {input}
Thought: {agent_scratchpad}"""

# Assicuriamoci che LangChain veda le variabili corrette
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
    max_iterations=5 # Per evitare loop infiniti
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