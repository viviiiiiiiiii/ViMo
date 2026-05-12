"""Calcola l'accuratezza delle risposte (Metriche) leggendo i file JSONL prodotti."""
import json, re, string
import evqa_eval_config as cfg

def clean_text(text: str) -> str:
    """Pulisce il testo: minuscolo, senza punteggiatura, senza articoli."""
    text = str(text or "").lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Rimuove gli articoli comuni
    text = re.sub(r"\b(a|an|the|il|lo|la|i|gli|le|un|uno|una)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def check_match(prediction: str, gold_answers: list[str]) -> tuple[bool, bool]:
    """Restituisce (Exact_Match, Relaxed_Match)."""
    pred_clean = clean_text(prediction)
    if not pred_clean: return False, False
    
    for gold in gold_answers:
        gold_clean = clean_text(gold)
        if not gold_clean: continue
        
        if pred_clean == gold_clean:
            return True, True  # Exact Match implica anche Relaxed Match
        if gold_clean in pred_clean or pred_clean in gold_clean:
            return False, True # Relaxed match (contenuto uno nell'altro)
            
    return False, False

def main():
    print(f"📊 Calcolando metriche su: {cfg.PREDICTIONS_PATH.name}")
    with open(cfg.PREDICTIONS_PATH) as f:
        rows = [json.loads(line) for line in f]

    stats = {"evaluated": 0, "errors": 0, "exact": 0, "relaxed": 0, "latency":[]}

    for row in rows:
        if row.get("error"): stats["errors"] += 1
        if "latency_sec" in row: stats["latency"].append(row["latency_sec"])
        
        golds = row.get("answers") or [row.get("gold_answer")]
        golds = [str(g) for g in golds if g]
        if not golds: continue

        stats["evaluated"] += 1
        exact, relaxed = check_match(row.get("prediction", ""), golds)
        if exact: stats["exact"] += 1
        if relaxed: stats["relaxed"] += 1

    # Calcolo percentuali finali
    n = stats["evaluated"] or 1 # Evita divisione per zero
    metrics = {
        "file": str(cfg.PREDICTIONS_PATH.name),
        "total_rows": len(rows),
        "errors": stats["errors"],
        "exact_match_accuracy": round((stats["exact"] / n) * 100, 2),
        "relaxed_match_accuracy": round((stats["relaxed"] / n) * 100, 2),
        "avg_latency_sec": round(sum(stats["latency"]) / len(stats["latency"]), 2) if stats["latency"] else 0
    }

    with open(cfg.METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Metriche Calcolate:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()