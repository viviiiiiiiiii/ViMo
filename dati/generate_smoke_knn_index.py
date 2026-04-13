
from pathlib import Path
import json
import numpy as np

try:
    import faiss
except ModuleNotFoundError as e:
    raise SystemExit(
        "Manca faiss. Installa prima una delle due opzioni:\n"
        "  pip install faiss-cpu\n"
        "oppure usa il tuo requirements del progetto."
    )

BASE = Path(__file__).resolve().parent
INDEX_JSON_PATH = BASE / "knn.json"
OUT_INDEX_PATH = BASE / "knn.index"

def main():
    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        index_map = json.load(f)

    dim = 512
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(len(index_map), dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(OUT_INDEX_PATH))

    print(f"Creato indice smoke-test: {OUT_INDEX_PATH}")
    print(f"Documenti indicizzati: {len(index_map)}")
    print("ATTENZIONE: questo indice è solo per verificare che la pipeline parta.")
    print("Non produce retrieval semantico reale.")

if __name__ == "__main__":
    main()
