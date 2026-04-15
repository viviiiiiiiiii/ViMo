from pathlib import Path
import json
import numpy as np
import torch
import faiss
from transformers import AutoModel, CLIPImageProcessor

from Qwen_retrieval import extract_features

BASE = Path(__file__).resolve().parent
KB_PATH = BASE / "encyclopedic_kb_wiki.json"
INDEX_JSON_PATH = BASE / "knn.json"
OUT_INDEX_PATH = BASE / "knn.index"
MODEL_NAME = "modelli/EVA-CLIP-8B"

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