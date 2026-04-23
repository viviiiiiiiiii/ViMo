

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
    """L'Adattatore che fa parlare LangChain con Qwen-VL"""
    
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # 1. TRUCCO MULTIMODALE
        match = re.search(r"L'immagine si trova in: '(.*?)'", prompt)
        image_path = match.group(1).strip("'\" ") if match else None

        # 2. Preparazione Messaggi
        user_content = []
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Sei un agente intelligente. Segui il formato Thought/Action/Observation."}]},
            {"role": "user", "content": user_content}
        ]

        # 📍 FIX PUNTO 2: Accesso tramite il modulo tools_real
        # Questo garantisce di leggere il modello caricato da start_motors()
        if tools_real.qwen_model is None or tools_real.qwen_processor is None:
            raise ValueError("Errore: I motori del server non sono stati accesi! Chiama start_motors prima di invocare l'agente.")

        risposta_grezza = generate_answer(
            tools_real.qwen_model, 
            tools_real.qwen_processor, 
            messages
        )

        # 4. IL FRENO A MANO
        if stop:
            for s in stop:
                if s in risposta_grezza:
                    risposta_grezza = risposta_grezza.split(s)[0]
                    
        return risposta_grezza.strip()

# ==========================================
# SETUP AGENTE (Globali)
# ==========================================
template_istruzioni = """Sei un assistente intelligente. Hai a disposizione i seguenti strumenti:
{tools}
Per usare uno strumento usa questo formato:
Thought: Devo capire cosa fare
Action: il nome dello strumento (deve essere uno tra {tool_names})
Action Input: l'input per lo strumento
Observation: il risultato dello strumento

Quando hai la risposta finale usa questo formato:
Thought: Ora so la risposta.
Final Answer: La tua risposta finale all'utente.

Inizia!
Domanda: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template_istruzioni)
vero_qwen = QwenServerLLM()

# Usiamo i riferimenti al modulo tools_real
agente = create_react_agent(vero_qwen, tools_real.miei_tools_reali, prompt)
esecutore = AgentExecutor(agent=agente, tools=tools_real.miei_tools_reali, verbose=True)

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
    try:
        immagine_base64 = image_to_base64(percorso_immagine)
    except FileNotFoundError:
        print("⚠️ File 'foto_buia.jpg' non trovato. Uso Base64 finto.")
        immagine_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    input_multimodale = [
        {"type": "text", "text": f"L'immagine si trova in: '{percorso_immagine}'. Chi ha dipinto questo quadro?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{immagine_base64}"}}
    ]

    print("\n🤖 Agente in ascolto...")
    esecutore.invoke({"input": input_multimodale})