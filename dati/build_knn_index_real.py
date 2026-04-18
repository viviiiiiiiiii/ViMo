import os
os.environ["TRANSFORMERS_IGNORE_LOAD_VULNERABILITY"] = "1"
import sys
from pathlib import Path
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# 📍 CONFIGURAZIONE PERCORSI
BASE_DATI = Path(__file__).resolve().parent
ROOT_VIMO = BASE_DATI.parent
sys.path.append(str(ROOT_VIMO)) # Permette di importare Qwen_retrieval

# Importiamo la logica di estrazione dall'Agente
from Qwen_retrieval import extract_features

# 📍 PUNTIAMO AL MODELLO NELLA TUA FOTO
MODEL_NAME = str(ROOT_VIMO / "modelli" / "EVA-CLIP-8B")
KB_PATH = BASE_DATI / "encyclopedic_kb_wiki.json"
INDEX_JSON_PATH = BASE_DATI / "knn.json"
OUT_INDEX_PATH = BASE_DATI / "knn.index"

def main():
    # 1. Caricamento dati
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        index_map = json.load(f)

    # 2. Caricamento Modello CLIP Locale
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🔄 Caricamento EVA-CLIP da: {MODEL_NAME}...")

        # Nel file build_knn_index_real.py, dopo 'model = AutoModel.from_pretrained(...)'
    if hasattr(model.config, "max_position_embeddings"):
        print(f"📏 LIMITE REALE DEL MODELLO: {model.config.max_position_embeddings} token")
    else:
        # In alcuni modelli EVA è dentro la config del text_config
        text_limit = getattr(model.config, "text_config", {}).get("max_position_embeddings", "Sconosciuto")
        
    print(f"📏 LIMITE TESTUALE: {text_limit} token")
    # Usiamo AutoProcessor per gestire sia immagini che testi
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
        trust_remote_code=True
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Preparazione testi
    texts = []
    for item in index_map:
        doc_id = item[0]
        entry = kb[doc_id]
        merged = entry.get("title", "") + " " + " ".join(entry.get("section_texts", []))
        texts.append(merged.strip())

    # 3. Estrazione Vettori (CLIP)
    all_embs = []
    print(f"🚀 Generazione embedding per {len(texts)} documenti...")
    
    for t in tqdm(texts):
        emb = extract_features(
            text=[t], 
            model=model, 
            processor=processor
        )
        all_embs.append(emb)

    # 4. Creazione Indice FAISS
    vecs = np.vstack(all_embs).astype("float32")
    
    # Usiamo IndexFlatIP per la Cosine Similarity (standard con CLIP)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    
    faiss.write_index(index, str(OUT_INDEX_PATH))

    print(f"\n✅ INDICE CREATO CON SUCCESSO!")
    print(f"📍 Percorso: {OUT_INDEX_PATH}")
    print(f"📊 Dimensione Vettori: {vecs.shape[1]} (Deve essere 512, 768 o 1024)")

if __name__ == "__main__":
    main()