
from pathlib import Path
import json
import numpy as np
import torch

try:
    import faiss
except ModuleNotFoundError as e:
    raise SystemExit(
        "Manca faiss. Installa prima una delle due opzioni:\n"
        "  pip install faiss-cpu\n"
        "oppure usa il tuo requirements del progetto."
    )

from transformers import AutoModel, AutoTokenizer

"""
Base per costruire un knn.index reale.

Questa versione indicizza il KB come TESTO.
Se il tuo progetto usa EVA-CLIP in modo multimodale, devi adattare la parte
di embedding al retriever reale che avete deciso di usare.
"""

BASE = Path(__file__).resolve().parent
KB_PATH = BASE / "encyclopedic_kb_wiki.json"
INDEX_JSON_PATH = BASE / "knn.json"
OUT_INDEX_PATH = BASE / "knn.index"

MODEL_NAME = "modelli/EVA-CLIP-8B"

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def main():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        index_map = json.load(f)

    texts = []
    for item in index_map:
        doc_id = item[0]
        entry = kb[doc_id]
        merged = entry.get("title", "") + "\\n" + "\\n".join(entry.get("section_texts", []))
        texts.append(merged.strip())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    all_embs = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            out = model(**toks)
            emb = mean_pool(out.last_hidden_state, toks["attention_mask"])
            all_embs.append(emb.cpu().numpy().astype("float32"))

    vecs = np.vstack(all_embs).astype("float32")
    vecs = l2_normalize(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(OUT_INDEX_PATH))

    print(f"Creato indice reale: {OUT_INDEX_PATH}")
    print(f"Documenti indicizzati: {len(texts)}")
    print(f"Dimensione embedding: {vecs.shape[1]}")

if __name__ == "__main__":
    main()
