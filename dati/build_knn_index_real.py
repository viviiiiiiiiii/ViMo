import os
os.environ["TRANSFORMERS_IGNORE_LOAD_VULNERABILITY"] = "1"
import sys
from pathlib import Path
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm

# 📍 CONFIGURAZIONE PERCORSI
BASE_DATI = Path(__file__).resolve().parent
ROOT_VIMO = BASE_DATI.parent
sys.path.append(str(ROOT_VIMO)) # Permette di importare Qwen_retrieval

# Importiamo la logica di estrazione e caricamento dall'Agente
from Qwen_retrieval import extract_features, load_clip_and_index

# 📍 DEFINIZIONE PERCORSI (Presi dal tuo config.json)
MODEL_NAME = str(ROOT_VIMO / "modelli" / "clip-vit-large-patch14")
KB_PATH = BASE_DATI / "encyclopedic_kb_wiki.json"
INDEX_JSON_PATH = BASE_DATI / "knn.json"
OUT_INDEX_PATH = BASE_DATI / "knn.index"

def main():
    # 1. Caricamento dati
    print(f"📂 Caricamento database Wikipedia da: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        index_map = json.load(f)

    # 2. Caricamento Modello CLIP con RIPARAZIONE RAM
    class FakeArgs:
        retriever_path = MODEL_NAME
        index_path = OUT_INDEX_PATH
        index_json_path = INDEX_JSON_PATH
        kb_wikipedia_path = KB_PATH
    
    print(f"🔄 Avvio procedura di riparazione e caricamento modello...")
    # Usiamo la funzione load_clip_and_index che abbiamo corretto per sanare i position_ids

    model, processor, _, _, _ = load_clip_and_index(FakeArgs(), load_faiss=False)    
    print(f"✅ Modello caricato su {model.device} e pronto per l'indicizzazione.")

    # 3. Preparazione testi
    texts = []
    for item in index_map:
        doc_id = item[0]
        entry = kb[doc_id]
        # Uniamo titolo e sezioni per creare un'impronta testuale ricca
        merged = entry.get("title", "") + " " + " ".join(entry.get("section_texts", []))
        texts.append(merged.strip())

    # 4. Estrazione Vettori (CLIP)
    all_embs = []
    print(f"🚀 Generazione embedding per {len(texts)} documenti...")
    
    for t in tqdm(texts):
        # Usiamo extract_features senza out_dim per mantenere le 1280 dimensioni originali
        emb = extract_features(
            text=t, 
            model=model, 
            processor=processor,
            out_dim=None 
        )
        all_embs.append(emb)

    # 5. Creazione Indice FAISS
    # Impiliamo i vettori in un'unica matrice numpy
    vecs = np.vstack(all_embs).astype("float32")
    
    print(f"📊 Dimensione finale vettori: {vecs.shape[1]}")
    
    # Creazione indice per similarità coseno (IndexFlatIP)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    
    # Salvataggio su disco
    faiss.write_index(index, str(OUT_INDEX_PATH))

    print(f"\n✅ DATABASE RIFATTO DA ZERO E SANATO!")
    print(f"📍 Nuovo file creato in: {OUT_INDEX_PATH}")
    print(f"📊 Numero documenti indicizzati: {len(texts)}")

if __name__ == "__main__":
    main()
