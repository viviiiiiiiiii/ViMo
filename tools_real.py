import torch
from PIL import Image
import ast
from langchain_core.tools import tool
from Qwen_retrieval import extract_features, retrieve_topk_pages, load_clip_and_index

import traceback

# sul server, in questo file dovrete caricare in memoria
clip_model = None
clip_processor = None
knn_index_immagini = None
knn_index_testi = None
wiki_map = None
wiki_data = None
qwen_model = None
qwen_processor = None

def start_motors(args):
    global clip_model, clip_processor, knn_index_immagini, knn_index_testi, wiki_map, wiki_data
    global qwen_model, qwen_processor 
    
    print("Accensione CLIP e FAISS...")
    clip_model, clip_processor, knn_index_immagini, wiki_map, wiki_data = load_clip_and_index(args)
    knn_index_testi = knn_index_immagini 
    
    print("Accensione Cervello Qwen2.5-VL...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    qwen_processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda:0"
    ).eval() 
    
    print("✅ motors good to goo")



import os
import ast
from PIL import Image

# ==============================================================================
# TOOL 1: LA RICERCA VISIVA REALE (MODIFICATO E BLINDATO)
# ==============================================================================
@tool
def tool_ricerca_visiva(image_path: str) -> str:
    """Usa questo tool PRIMA DI TUTTO se l'utente fornisce un'immagine. 
    Passagli solo il percorso dell'immagine (es. 'foto_buia.jpg')."""
    
    # 📍 PULIZIA 1: Se l'agente manda un dizionario stringato {'image_path': '...'}
    try:
        clean_path = image_path.strip()
        if clean_path.startswith("{"):
            parsed = ast.literal_eval(clean_path)
            image_path = parsed.get("image_path", clean_path)
    except:
        pass
    
# 📍 PULIZIA 2: Il trucco salva-vita potenziato
    image_path = str(image_path).strip().split('\n')[0].replace('"', '').replace("'", "")
    
    # 📍 NOVITÀ: Rimuoviamo il prefisso se il modello fa il precisino
    if image_path.startswith("image_path="):
        image_path = image_path.replace("image_path=", "").strip()
    elif image_path.startswith("image="):
        image_path = image_path.replace("image=", "").strip()
    
    print(f"\n[TOOL VISIVO] Sto analizzando i pixel di: {image_path}")
    
    # 📍 PROTEZIONE FILE: Prima di far esplodere PIL, verifichiamo che il file esista
    if not os.path.exists(image_path):
        return f"Errore: Il file '{image_path}' non esiste nel server. Assicurati di non aver inventato il nome."
    
    try:
        # 1. Apertura immagine
        image_pil = Image.open(image_path).convert("RGB")
        
        # 2. Estrazione feature con CLIP
        image_features = extract_features(
            image=image_pil, 
            text=None, 
            model=clip_model, 
            processor=clip_processor, 
            out_dim=512
        )
        
        # 3. Ricerca nel database
        testi_enciclopedia = retrieve_topk_pages(
            features=image_features, 
            index=knn_index_immagini, 
            index_map=wiki_map, 
            wiki=wiki_data, 
            k=1
        )
        
        return f"Contesto trovato dal database visivo:\n{testi_enciclopedia}"

    except Exception as e:
        # 📍 IL RILEVATORE DI BUG: Stampa l'errore completo nel log di Slurm
        print("\n=== DETTAGLIO ERRORE VISIVO ===")
        traceback.print_exc()
        print("===============================\n")
        
        return f"Errore nel database visivo: {str(e)}. Prova a usare la ricerca testuale."


# ==============================================================================
# TOOL 2: LA RICERCA TESTUALE REALE (MODIFICATO E BLINDATO)
# ==============================================================================
@tool
def tool_ricerca_testuale(query: str) -> str:
    """Usa questo tool come SECONDA OPZIONE o per raffinare la ricerca.
    Passagli solo parole chiave (es. 'Leonardo da Vinci'), mai immagini."""
    
    # 📍 PULIZIA 1: Dizionari stringati
    try:
        clean_query = query.strip()
        if clean_query.startswith("{"):
            parsed = ast.literal_eval(clean_query)
            query = parsed.get("query", clean_query)
    except:
        pass
        
# 📍 PULIZIA 2 per il testo
    query = str(query).strip().split('\n')[0].replace('"', '').replace("'", "")
    
    # 📍 NOVITÀ: Rimuoviamo il prefisso
    if query.startswith("query="):
        query = query.replace("query=", "").strip()
    elif query.startswith("testo="):
        query = query.replace("testo=", "").strip()
    
    print(f"\n[TOOL TESTUALE] Sto cercando le parole chiave: '{query}'")

    try:
        # Estrazione feature testuali
        text_features = extract_features(
            image=None, 
            text=query, 
            model=clip_model, 
            processor=clip_processor, 
            out_dim=512
        )
        
        # Ricerca nel database
        testi_enciclopedia = retrieve_topk_pages(
            features=text_features, 
            index=knn_index_testi, # <-- Assicurati che esista questo index per il testo!
            index_map=wiki_map, 
            wiki=wiki_data, 
            k=3
        )
        
        return f"Contesto trovato dal database testuale:\n{testi_enciclopedia}"

    except Exception as e:
        return f"Errore nel database testuale: {str(e)}"

# Li impacchettiamo per LangChain
miei_tools_reali = [tool_ricerca_visiva,tool_ricerca_testuale]