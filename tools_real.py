import torch
from PIL import Image
from langchain_core.tools import tool
from Qwen_retrieval import extract_features, retrieve_topk_pages, load_clip_and_index

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
        trust_remote_code=True
    ).to(device).eval() 
    
    print("✅ motors good to goo")


# ==============================================================================
# TOOL 1: LA RICERCA VISIVA REALE
# ==============================================================================
@tool
def tool_ricerca_visiva(image_path: str) -> str:
    """Usa questo tool PRIMA DI TUTTO se l'utente fornisce un'immagine. 
    Passagli il percorso dell'immagine (es. 'foto_buia.jpg') per cercare nel database visivo.
    Se l'immagine è sfuocata o non trovi la risposta, prova a usare la ricerca testuale."""
    
    print(f"\n[TOOL VISIVO] Sto analizzando i pixel di: {image_path}")
    
    try:
        # 1. Apriamo l'immagine fisicamente
        image_pil = Image.open(image_path).convert("RGB")
        
        # 2.Passiamo i pixel a CLIP
        image_features = extract_features(
            image=image_pil, 
            text=None, 
            model=clip_model, 
            processor=clip_processor, 
            out_dim=512
        )
        
        # 3.Cerchiamo i vettori più simili
        testi_enciclopedia = retrieve_topk_pages(
            features=image_features, 
            knn_index=knn_index_immagini, 
            knn_map=wiki_map, 
            wiki=wiki_data, 
            k=3 # Prendiamo i 3 risultati migliori
        )
        
        return f"Contesto trovato dal database visivo:\n{testi_enciclopedia}"

    except Exception as e:
        # Se qualcosa va storto (es. file non trovato), l'Agente non crasherà, 
        return f"Errore nel database visivo: {str(e)}. Prova a usare la ricerca testuale."


# ==============================================================================
# TOOL 2: LA RICERCA TESTUALE REALE
# ==============================================================================
@tool
def tool_ricerca_testuale(query: str) -> str:
    """Usa questo tool come SECONDA OPZIONE, o se vuoi raffinare la ricerca usando parole chiave.
    Non passargli immagini, ma solo stringhe di testo."""
    
    print(f"\n[TOOL TESTUALE] Sto cercando le parole chiave: '{query}'")

    
    try:
        # passiamo la stringa di testo a CLIP (Text Encoder)
        text_features = extract_features(
            image=None, 
            text=[query], 
            model=clip_model, 
            processor=clip_processor, 
            out_dim=512
        )
        
        # Cerchiamo nel database dei testi
        testi_enciclopedia = retrieve_topk_pages(
            features=text_features, 
            knn_index=knn_index_testi, 
            knn_map=wiki_map, 
            wiki=wiki_data, 
            k=3
        )
        
        return f"Contesto trovato dal database testuale:\n{testi_enciclopedia}"

    except Exception as e:
        return f"Errore nel database testuale: {str(e)}"

# Li impacchettiamo per LangChain
miei_tools_reali = [tool_ricerca_visiva, tool_ricerca_testuale]