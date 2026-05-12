"""Valuta il sistema RAG normale sul subset EVQA creato in precedenza."""
import json, time
import evqa_eval_config as cfg
from rag_normal_real import build_args, load_rag_engines, normal_rag_answer

def load_done_ids(filepath) -> set:
    """Legge le domande a cui abbiamo già risposto per non ripeterle."""
    if not filepath.exists(): return set()
    with open(filepath) as f:
        return {json.loads(line)["id"] for line in f if "id" in json.loads(line)}

def main():
    with open(cfg.SUBSET_OUTPUT_FILE) as f:
        examples = [json.loads(line) for line in f]
    if cfg.EVAL_LIMIT: examples = examples[:cfg.EVAL_LIMIT]

    cfg.RAG_NORMAL_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(cfg.RAG_NORMAL_PREDICTIONS_FILE, "a", encoding="utf-8")
    
    done_ids = load_done_ids(cfg.RAG_NORMAL_PREDICTIONS_FILE) if cfg.RESUME_IF_OUTPUT_EXISTS else set()

    print("🔧 Caricamento motori RAG in corso...")
    engines = load_rag_engines(build_args(top_k=cfg.TOP_K))

    for i, ex in enumerate(examples, 1):
        if ex["id"] in done_ids: continue

        print(f"\n[{i}/{len(examples)}] Q: {ex['question']}")
        start_time = time.time()
        
        # Prepariamo l'output clonando la domanda originale
        out = ex.copy()
        out["top_k"] = cfg.TOP_K
        
        try:
            result = normal_rag_answer(ex["question"], ex.get("image_path"), cfg.TOP_K, engines)
            out["prediction"] = result.get("answer", "")
            out["error"] = None
            if cfg.SAVE_RETRIEVED_CONTEXT: out["retrieved_context"] = result.get("context", "")
        except Exception as e:
            out["prediction"] = ""
            out["error"] = str(e)
            print(f"❌ Errore: {e}")

        out["latency_sec"] = round(time.time() - start_time, 3)
        
        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
        f_out.flush()
        print(f"🤖 Risp: {out['prediction'][:150]}...")

    f_out.close()
    print("✅ Valutazione RAG Normale completata!")

if __name__ == "__main__":
    main()