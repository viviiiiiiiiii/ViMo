import sys
from pathlib import Path

# 📍 FIX PERCORSI: Aggiungiamo la cartella radice (VIMO) al cammino di Python
BASE_DATI = Path(__file__).resolve().parent
ROOT_VIMO = BASE_DATI.parent
sys.path.append(str(ROOT_VIMO))

import json
import numpy as np
import torch
import faiss

# Ora questo funzionerà perché abbiamo aggiunto ROOT_VIMO al path!
from Qwen_retrieval import extract_features

# 📍 FIX MODELLO: Punta alla cartella modelli che è sorella di dati
MODEL_NAME = str(ROOT_VIMO / "modelli" / "EVA-CLIP-8B")
KB_PATH = BASE_DATI / "encyclopedic_kb_wiki.json"
INDEX_JSON_PATH = BASE_DATI / "knn.json"
OUT_INDEX_PATH = BASE_DATI / "knn.index"

def main():
    # 1. Caricamento dati (identico a prima)
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        index_map = json.load(f)

    # 2. Caricamento Modello (Usiamo le stesse impostazioni del retriever)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16 if device == "cuda:0" else torch.float32,
        trust_remote_code=True
    ).to(device).eval()
    
    # Per il testo CLIP usa il tokenizer, ma noi usiamo il processor che lo include
    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME) 

    texts = []
    for item in index_map:
        doc_id = item[0]
        entry = kb[doc_id]
        # Pulizia testo
        merged = entry.get("title", "") + " " + " ".join(entry.get("section_texts", []))
        texts.append(merged.strip())

    # 3. ESTRAZIONE VETTORI (Usando la funzione condivisa!)
    all_embs = []
    print(f"Generazione embedding per {len(texts)} documenti...")
    
    for t in texts:
        # Usiamo extract_features di Qwen_retrieval per coerenza totale!
        emb = extract_features(
            image=None, 
            text=[t], 
            model=model, 
            processor=processor
        )
        all_embs.append(emb)

    vecs = np.vstack(all_embs).astype("float32")

    # 4. CREAZIONE INDICE FAISS
    # IndexFlatIP + Normalizzazione = Cosine Similarity (perfetto per CLIP)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    
    faiss.write_index(index, str(OUT_INDEX_PATH))

    print(f"✅ Indice sincronizzato creato: {OUT_INDEX_PATH}")
    print(f"Dimensione vettori: {vecs.shape[1]}") # Sarà 512 o 768 a seconda di EVA-CLIP

if __name__ == "__main__":
    main()