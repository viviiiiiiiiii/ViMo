import os
os.environ["TRANSFORMERS_IGNORE_LOAD_VULNERABILITY"] = "1"
import sys
from pathlib import Path
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm

# 📍 CONFIGURAZIONE PERCORSI PER COLAB
# Prende la directory corrente (/content/drive/MyDrive/ViMo)
BASE_DIR = Path.cwd() 
DATI_DIR = BASE_DIR / "dati"
sys.path.append(str(BASE_DIR)) # Permette di importare Qwen_retrieval

from Qwen_retrieval import extract_features, load_clip_and_index

# 📍 DEFINIZIONE PERCORSI (MODELLO ONLINE)
MODEL_NAME = "openai/clip-vit-large-patch14"
KB_PATH = DATI_DIR / "encyclopedic_kb_wiki.json"
INDEX_JSON_PATH = DATI_DIR / "knn.json"
OUT_INDEX_PATH = DATI_DIR / "knn.index"

def main():
    # 1. Caricamento dati
    print(f"📂 Caricamento database Wikipedia da: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        index_map = json.load(f)

    # 2. Caricamento Modello CLIP (Online)
    class FakeArgs:
        retriever_path = MODEL_NAME
        index_path = OUT_INDEX_PATH
        index_json_path = INDEX_JSON_PATH
        kb_wikipedia_path = KB_PATH
    
    print(f"🔄 Avvio caricamento modello CLIP...")
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
        # Usiamo extract_features senza out_dim per mantenere le dimensioni originali
        emb = extract_features(
            text=t, 
            model=model, 
            processor=processor,
            out_dim=None 
        )
        all_embs.append(emb)

    # 5. Creazione Indice FAISS
    vecs = np.vstack(all_embs).astype("float32")
    print(f"📊 Dimensione finale vettori: {vecs.shape[1]}")
    
    # Creazione indice per similarità (Prodotto Interno = Coseno se normalizzati)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    
    # Salvataggio su disco (Direttamente nel tuo Drive)
    faiss.write_index(index, str(OUT_INDEX_PATH))

    print(f"\n✅ DATABASE RIFATTO DA ZERO E SALVATO SU DRIVE!")
    print(f"📍 Nuovo file creato in: {OUT_INDEX_PATH}")
    print(f"📊 Numero documenti indicizzati: {len(texts)}")

if __name__ == "__main__":
    main()