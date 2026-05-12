

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

        # 📍 AGGIUNGIAMO PARAMETRI ANTI-LOOP
        # Nota: assicurati che generate_answer accetti **kwargs o parametri extra
        risposta = generate_answer(
            tools_real.qwen_model, 
            tools_real.qwen_processor, 
            messages,
            temperature=0.1,         # Più basso = meno fantasia
            repetition_penalty=1.2,  # 📍 BLOCCA I LOOP DI RIPETIZIONE
            max_new_tokens=512       # Evita risposte infinite
        )

        if stop is not None:
            for stop_word in stop:
                if stop_word in risposta:
                    risposta = risposta.split(stop_word)[0]

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

# ==========================================
# SETUP AGENTE (Globali) - VERSIONE CORRETTA
# ==========================================

template_universale = """You are a DATA-ONLY research assistant. 
You must identify subjects and then verify details ONLY using the provided tools.

You have access to the following tools:
{tools}

RULES:
1. NEVER invent information. If it's not in the 'Observation', it doesn't exist.
2. If you identify a subject (e.g., Leonardo) but the user asks for details (e.g., inventions) that are NOT in the visual observation, you MUST call 'tool_ricerca_testuale' before answering.
3. If BOTH tools fail to provide specific info, say: "The database does not contain information about [X]".
4. Do not repeat yourself.

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

(this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer
Final Answer: [Summarize ONLY what was found in the observations]

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
# FUNZIONI PER LA EVALUATION AUTOMATICA
# ==========================================

def build_args(top_k=3):
    """Costruisce gli argomenti per l'agente durante la valutazione."""
    config_dict = load_config()
    class CostruttoreArgs: pass
    args = CostruttoreArgs()
    for key, value in config_dict.items():
        setattr(args, key, str(value))
    args.top_k = top_k
    return args

def load_agentic_engines(args):
    """Accende i motori globali in tools_real una sola volta per tutte le 1000 domande."""
    print("Accensione motori globali dell'agente in corso...")
    tools_real.start_motors(args)
    return True # Ritorna un check, i modelli sono salvati in tools_real

def agentic_rag_answer(question: str, image_path: Optional[str] = None, top_k: int = 3, engines=None):
    """La funzione che la pipeline di valutazione chiama per ogni domanda."""
    
    # Se c'è un'immagine, la uniamo testualmente alla domanda in modo che l'agente lo sappia
    if image_path:
        prompt_completo = f"Immagine fornita: '{image_path}'.\nDomanda: {question}"
    else:
        prompt_completo = f"Domanda: {question}"

    # Eseguiamo l'agente
    risultato = esecutore.invoke({"input": prompt_completo})
    
    # LangChain AgentExecutor di solito restituisce la risposta finale nella chiave "output"
    return risultato.get("output", str(risultato))


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