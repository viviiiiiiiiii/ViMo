import base64
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.llms.fake import FakeListLLM

# ==========================================
# FUNZIONE DI SUPPORTO: DA IMMAGINE A TESTO
# ==========================================
def image_to_base64(image_path):
    # Apre l'immagine in binario, la codifica e la trasforma in stringa
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# ==========================================
# I DUE TOOL (Con lo "spazio" per CLIP)
# ==========================================
@tool
def tool_ricerca_visiva(image_path: str) -> str:
    """Usa questo tool PRIMA DI TUTTO se l'utente fornisce un'immagine. 
    Passagli il percorso dell'immagine (es. 'foto_buia.jpg') per cercare nel database.
    Se l'immagine è sfuocata o non trovi la risposta, prova a usare la ricerca testuale."""
    
    print(f"\n👀 [AZIONE LANCIATA] Eseguo ricerca visiva per: {image_path}")
    
    # -------------------------------------------------------------------------
    # 📍 IL PUNTO ESATTO IN CUI ENTRANO IN GIOCO CLIP E FAISS (Nella Versione Reale)
    # -------------------------------------------------------------------------
    #
    # 1. Carica l'immagine dal percorso che Qwen gli ha passato:
    #    image_pil = Image.open(image_path).convert("RGB")
    #
    # 2. Passa l'immagine all'Image Encoder di CLIP per ottenere il vettore:
    #    image_features = extract_features(image_pil, clip_model, clip_processor, 512)
    #
    # 3. Cerca il vettore in FAISS e recupera i testi enciclopedici:
    #    risultato_wiki = retrieve_topk_pages(image_features, knn_index, knn_map, wiki_data, k=3)
    #
    # 4. Invece della stringa finta qui sotto, restituirai 'risultato_wiki'
    # -------------------------------------------------------------------------
    
    # Per ora simuliamo un fallimento del database
    return "Contesto trovato: Immagine molto buia, sembra una cornice di legno del 1400. Nessun pittore specificato."

@tool
def tool_ricerca_testuale(query: str) -> str:
    """Usa questo tool come SECONDA OPZIONE, o se vuoi raffinare la ricerca usando parole chiave.
    Non passargli immagini, ma solo stringhe di testo."""
    
    print(f"\n📖 [AZIONE LANCIATA] Eseguo ricerca testuale per: {query}")
    return "Contesto trovato: Il dipinto noto come La Vergine delle Rocce è stato realizzato da Leonardo da Vinci."

miei_tools = [tool_ricerca_visiva, tool_ricerca_testuale]

# ==========================================
# IL PROMPT E L'AGENTE
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

risposte_finte = [
    "Thought: Devo usare l'immagine.\nAction: tool_ricerca_visiva\nAction Input: foto_buia.jpg",
    "Thought: L'immagine non ha aiutato. Provo col testo.\nAction: tool_ricerca_testuale\nAction Input: Vergine delle Rocce pittore",
    "Thought: Ora lo so.\nFinal Answer: L'ha dipinto Leonardo da Vinci."
]
finto_qwen = FakeListLLM(responses=risposte_finte)

agente = create_react_agent(finto_qwen, miei_tools, prompt)
esecutore = AgentExecutor(agent=agente, tools=miei_tools, verbose=True)

# ==========================================
# L'ESECUZIONE CON INPUT MULTIMODALE
# ==========================================
if __name__ == "__main__":
    # ATTENZIONE: Crea un file vuoto o metti un'immagine vera chiamata 'foto_buia.jpg' 
    # nella stessa cartella dello script, altrimenti image_to_base64 andrà in errore!
    percorso_immagine = "foto_buia.jpg" 
    
    # Usiamo un try/except per farti testare il codice anche se non hai ancora creato l'immagine finta
    try:
        immagine_base64 = image_to_base64(percorso_immagine)
        print("✅ Immagine convertita in Base64 con successo!")
    except FileNotFoundError:
        print("⚠️ File 'foto_buia.jpg' non trovato. Uso una stringa Base64 finta per il test.")
        immagine_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    # COSTRUIAMO IL VERO INPUT PER QWEN-VL
    input_multimodale = [
        {
            "type": "text", 
            "text": f"L'immagine di riferimento si trova nel percorso: '{percorso_immagine}'. Chi ha dipinto questo quadro?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{immagine_base64}"
            }
        }
    ]

    print("\n🚀 Avvio l'Agente...")
    esecutore.invoke({"input": input_multimodale})