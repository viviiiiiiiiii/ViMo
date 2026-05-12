"""Valuta l'Agente RAG avanzato importandolo dinamicamente dal config."""
import json, time, importlib
import evqa_eval_config as cfg

def load_done_ids(filepath) -> set:
    if not filepath.exists(): return set()
    with open(filepath) as f:
        return {json.loads(line)["id"] for line in f if "id" in json.loads(line)}

def setup_agent():
    """Importa dinamicamente le funzioni dell'agente scritte nel config."""
    module = importlib.import_module(cfg.AGENT_MODULE)
    agent_fn = getattr(module, cfg.AGENT_FUNCTION)
    
    engines = None
    if cfg.AGENT_LOAD_ENGINES_FUNCTION and hasattr(module, cfg.AGENT_LOAD_ENGINES_FUNCTION):
        load_fn = getattr(module, cfg.AGENT_LOAD_ENGINES_FUNCTION)
        build_fn = getattr(module, cfg.AGENT_BUILD_ARGS_FUNCTION, lambda top_k: None)
        engines = load_fn(build_fn(top_k=cfg.TOP_K) or build_fn())
        
    return agent_fn, engines

def main():
    with open(cfg.SUBSET_OUTPUT_FILE) as f:
        examples = [json.loads(line) for line in f]
    if cfg.EVAL_LIMIT: examples = examples[:cfg.EVAL_LIMIT]

    cfg.AGENTIC_PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(cfg.AGENTIC_PREDICTIONS_FILE, "a", encoding="utf-8")
    done_ids = load_done_ids(cfg.AGENTIC_PREDICTIONS_FILE) if cfg.RESUME_IF_OUTPUT_EXISTS else set()

    print(f"🤖 Caricamento Agente {cfg.AGENT_MODULE} in corso...")
    agent_fn, engines = setup_agent()

    for i, ex in enumerate(examples, 1):
        if ex["id"] in done_ids: continue

        print(f"\n[{i}/{len(examples)}] Q: {ex['question']}")
        start_time = time.time()
        
        out = ex.copy()
        out["top_k"] = cfg.TOP_K
        
        try:
            # Chiamata flessibile all'agente
            res = agent_fn(question=ex["question"], image_path=ex.get("image_path"), top_k=cfg.TOP_K, engines=engines)
            
            # Estrae la risposta (che sia una stringa o un dizionario)
            out["prediction"] = res if isinstance(res, str) else res.get("answer", res.get("prediction", str(res)))
            out["error"] = None
        except Exception as e:
            out["prediction"] = ""
            out["error"] = str(e)
            print(f"❌ Errore: {e}")

        out["latency_sec"] = round(time.time() - start_time, 3)
        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
        f_out.flush()
        print(f"🤖 Risp: {out['prediction'][:150]}...")

    f_out.close()
    print("✅ Valutazione RAG Agente completata!")

if __name__ == "__main__":
    main()