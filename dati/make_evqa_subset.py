"""Prende il dataset completo, lo filtra e crea un subset di N domande."""
import ast, csv, json, random
from pathlib import Path
import evqa_eval_config as cfg

def read_dataset(path: Path) -> list[dict]:
    """Legge JSONL, JSON o CSV restituendo una lista di dizionari."""
    if path.suffix == ".jsonl":
        with open(path) as f: return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        with open(path) as f: 
            data = json.load(f)
            return data if isinstance(data, list) else data.get("data", data.get("examples",[]))
    if path.suffix == ".csv":
        with open(path) as f: return list(csv.DictReader(f))
    raise ValueError("Formato non supportato")

def clean_list(val) -> list[str]:
    """Trasforma stringhe sporche (es. \"['img1']\" o 'a|b') in vere liste Python."""
    if not val or str(val).lower() in ("nan", "none"): return[]
    if isinstance(val, (list, tuple)): return [str(x).strip() for x in val]
    
    val = str(val).strip()
    if val.startswith("["): # Prova a interpretare stringhe che sembrano liste
        try: return [str(x).strip() for x in ast.literal_eval(val)]
        except: pass
        
    # Divide per delimitatori strani usati in EVQA
    return [x.strip() for x in val.replace("|", "&&").replace(",", "&&").split("&&") if x.strip()]

def find_image(image_id: str, dataset_name: str) -> str | None:
    """Cerca l'immagine nelle cartelle locali usando varie estensioni."""
    if not image_id: return None
    
    cartelle = [cfg.IMAGE_ROOT / str(dataset_name), cfg.IMAGE_ROOT]
    estensioni = ["", ".jpg", ".jpeg", ".png", ".webp"]
    
    for cartella in cartelle:
        for est in estensioni:
            path = cartella / f"{image_id}{est}"
            if path.exists(): return str(path)
    return None

def main():
    print(f"📖 Leggendo {cfg.EVQA_TEST_FILE}...")
    dataset = read_dataset(cfg.EVQA_TEST_FILE)
    
    # Filtra solo domande single-hop (dirette)
    esclusi = {x.lower().replace("-", "_") for x in cfg.EXCLUDE_QUESTION_TYPES}
    single_hop = [row for row in dataset if str(row.get("question_type", "")).lower() not in esclusi]
    
    # Prende un campione casuale
    random.seed(cfg.RANDOM_SEED)
    subset = random.sample(single_hop, min(cfg.N_QUESTIONS, len(single_hop)))

    output_data =[]
    for i, row in enumerate(subset):
        answers = clean_list(row.get("answer") or row.get("answers"))
        image_ids = clean_list(row.get("dataset_image_ids", row.get("image_ids")))
        dataset_name = row.get("dataset_name", "")
        img_id = image_ids[0] if image_ids else None
        
        output_data.append({
            "id": row.get("id", f"row_{i}"),
            "question": row.get("question", ""),
            "answers": answers,
            "gold_answer": answers[0] if answers else None,
            "question_type": row.get("question_type", ""),
            "image_path": find_image(img_id, dataset_name),
            "wikipedia_title": row.get("wikipedia_title"),
        })

    cfg.SUBSET_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.SUBSET_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ Creato subset di {len(output_data)} domande in {cfg.SUBSET_OUTPUT_FILE.name}")

if __name__ == "__main__":
    main()