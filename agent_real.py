

import base64
import torch
import re
from PIL import Image
from typing import Optional, List

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
# L'ADATTATORE QWEN (Corretto per il Punto 2)
# ==========================================
class QwenServerLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # 📍 RENDIAMO TUTTO TESTUALE: L'agente tratta il percorso immagine come testo
        messages = [{"role": "user", "content": prompt}]

        if tools_real.qwen_model is None or tools_real.qwen_processor is None:
            raise ValueError("Motori non accesi!")

        return generate_answer(tools_real.qwen_model, tools_real.qwen_processor, messages)

# ==========================================
# SETUP AGENTE (Globali)
# ==========================================
template_istruzioni = """Sei un assistente intelligente. Hai a disposizione questi strumenti:
{tools}

Per usare uno strumento usa ESATTAMENTE questo formato:
Thought: Devo capire cosa fare
Action: il nome dello strumento (uno tra {tool_names})
Action Input: solo la stringa di testo o il percorso file (SENZA parentesi graffe o JSON)
Observation: il risultato dello strumento

Esempio di Action Input corretto: foto_buia.jpg
Esempio di Action Input ERRATO: {{'image_path': 'foto_buia.jpg'}}

Quando hai la risposta finale:
Thought: Ora so la risposta.
Final Answer: La tua risposta finale.

Inizia!
Domanda: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template_istruzioni)
vero_qwen = QwenServerLLM()

# Usiamo i riferimenti al modulo tools_real
agente = create_react_agent(vero_qwen, tools_real.miei_tools_reali, prompt)
esecutore = AgentExecutor(agent=agente, tools=tools_real.miei_tools_reali, verbose=True,handle_parsing_errors=True)

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
    input_semplice = f"L'immagine si trova in: '{percorso_immagine}'. Chi ha dipinto questo quadro?"

    print("\n🤖 Agente in ascolto (Stabile)...")
    esecutore.invoke({"input": input_semplice})