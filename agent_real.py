import base64
import torch
from PIL import Image
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.llms.fake import FakeListLLM
import load_config


from Qwen_retrieval import extract_features, retrieve_topk_pages
from tools_real import miei_tools_reali, start_motors

# ==========================================
# VARIABILI GLOBALI PER IL SERVER
# ==========================================
# Queste le de-commenterete quando sarete sul server per caricare i modelli pesanti
# clip_model = ... 
# clip_processor = ...
# knn_index_immagini = ... 
# knn_index_testi = ... 
# wiki_map = ... 
# wiki_data = ...

# ==========================================
# FUNZIONE DI SUPPORTO: DA IMMAGINE A TESTO
# ==========================================
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# ==========================================
# IL PROMPT E L'AGENTE (Intatti!)
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

# Per i test in locale , teniamo ancora il finto Qwen
risposte_finte = [
    "Thought: Devo usare l'immagine.\nAction: tool_ricerca_visiva\nAction Input: foto_buia.jpg",
    "Thought: L'immagine non ha aiutato. Provo col testo.\nAction: tool_ricerca_testuale\nAction Input: Vergine delle Rocce pittore",
    "Thought: Ora lo so.\nFinal Answer: L'ha dipinto Leonardo da Vinci."
]

#finto_qwen = FakeListLLM(responses=risposte_finte)

from typing import Optional, List
from langchain.llms.base import LLM
import re#?
from Qwen_retrieval import generate_answer

class QwenServerLLM(LLM):
    """L'Adattatore che fa parlare LangChain con Qwen-VL"""
    
    @property
    def _llm_type(self) -> str:
        return "qwen2.5-vl-custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # 1. TRUCCO MULTIMODALE: Cerchiamo il nome dell'immagine nel testo!
        match = re.search(r"L'immagine si trova in: '(.*?)'", prompt)
        image_path = match.group(1) if match else None

        # 2. Prepariamo il pacchetto esattamente come lo vuole il tuo amico
        user_content = []
        if image_path:
            user_content.append({"type": "image", "image": image_path})
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Sei un agente intelligente. Segui rigorosamente il formato Thought/Action/Observation."}]},
            {"role": "user", "content": user_content}
        ]

        # 3. FACCIAMO PENSARE QWEN! (Usando le variabili globali caricate prima)
        # Assicurati di importare qwen_model e qwen_processor dal file dove li hai definiti
        from tools_real import qwen_model, qwen_processor 
        risposta_grezza = generate_answer(qwen_model, qwen_processor, messages)

        # 4. IL FRENO A MANO (CRUCIALE PER GLI AGENTI REACT)
        # LangChain passerà stop=["Observation:"]. Se non tagliamo la risposta qui,
        # Qwen proverà a interpretare il ruolo del database e si inventerà i dati!
        if stop:
            for s in stop:
                if s in risposta_grezza:
                    # Taglia la stringa nel punto esatto in cui compare "Observation:"
                    risposta_grezza = risposta_grezza.split(s)[0]
                    
        return risposta_grezza.strip()

# ==========================================
# L'INIZIALIZZAZIONE FINALE
# ==========================================
# Creiamo il vero LLM
vero_qwen = QwenServerLLM()

# Creiamo l'Agente!
agente = create_react_agent(vero_qwen, miei_tools_reali, prompt)
esecutore = AgentExecutor(agent=agente, tools=miei_tools_reali, verbose=True)


#agente = create_react_agent(finto_qwen, miei_tools_reali, prompt)
#esecutore = AgentExecutor(agent=agente, tools=miei_tools_reali, verbose=True)

# ==========================================
# L'ESECUZIONE
# ==========================================
if __name__ == "__main__":
    print("Inizializzazione sistema...")
    
    # 1. LEGGIAMO I PERCORSI DEL SERVER 
    config_dict = load_config()

    class CostruttoreArgs:
        pass
    
    args = CostruttoreArgs()
    for key, value in config_dict.items():
        setattr(args, key, str(value))
    
    args.top_k=3
    # 2. GIRIAMO LA CHIAVE! (Questo caricherà i Gigabyte in memoria)
    start_motors(args)
    
    # 3. PREPARIAMO L'IMMAGINE DELL'UTENTE
    percorso_immagine = "foto_buia.jpg" 
    try:
        immagine_base64 = image_to_base64(percorso_immagine)
    except FileNotFoundError:
        print("⚠️ File non trovato. Uso Base64 finto.")
        immagine_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    input_multimodale = [
        {"type": "text", "text": f"L'immagine si trova in: '{percorso_immagine}'. Chi ha dipinto questo quadro?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{immagine_base64}"}}
    ]

    print("\n🚀 Avvio l'Agente Modulare...")
    
    # 4. VIA AL LOOP!
    esecutore.invoke({"input": input_multimodale})