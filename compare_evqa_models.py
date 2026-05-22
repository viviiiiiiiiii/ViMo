import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    for key in ["data", "examples", "questions", "samples", "items"]:
        if key in obj and isinstance(obj[key], list):
            return obj[key]

    raise ValueError(f"Formato JSON non riconosciuto: {path}")


def normalize_question_type(qtype: Any) -> str:
    qtype = str(qtype).strip().lower()

    mapping = {
        "multi-answer": "multi_answer",
        "multianswer": "multi_answer",
        "multi_answer": "multi_answer",
        "2-hop": "2_hop",
        "2hop": "2_hop",
        "2_hop": "2_hop",
        "automatic": "automatic",
        "templated": "templated",
    }

    return mapping.get(qtype, qtype)


def get_reference_list(sample: Dict[str, Any]) -> List[str]:
    if "answers" in sample and sample["answers"]:
        return [str(a) for a in sample["answers"]]

    answer = str(sample.get("answer", "")).strip()

    if not answer:
        return []

    refs = []
    for part in answer.split("|"):
        part = part.strip()
        if part and part not in refs:
            refs.append(part)

    return refs


def get_question_id(sample: Dict[str, Any], fallback_idx: int) -> str:
    for key in ["id", "question_id", "qid", "sample_id"]:
        if key in sample and sample[key] not in [None, ""]:
            return str(sample[key])
    return str(fallback_idx)


def get_question(sample: Dict[str, Any]) -> str:
    for key in ["question", "query", "text"]:
        if key in sample and sample[key] not in [None, ""]:
            return str(sample[key])
    return ""


def load_predictions(path: Path) -> Dict[str, str]:
    rows = load_json_or_jsonl(path)

    pred_by_id = {}

    for row in rows:
        if "id" in row:
            qid = str(row["id"])
        elif "question_id" in row:
            qid = str(row["question_id"])
        else:
            continue

        prediction = (
            row.get("prediction")
            or row.get("answer")
            or row.get("output")
            or row.get("response")
            or ""
        )

        pred_by_id[qid] = str(prediction)

    return pred_by_id


def load_records_optional(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}

    if not path.exists():
        return {}

    rows = load_json_or_jsonl(path)
    out = {}

    for row in rows:
        qid = row.get("question_id", row.get("id", None))
        if qid is not None:
            out[str(qid)] = row

    return out


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def evaluate_model_on_gold(
    *,
    model_name: str,
    gold: List[Dict[str, Any]],
    pred_by_id: Dict[str, str],
    records_by_id: Dict[str, Dict[str, Any]],
    evaluation_utils,
) -> Dict[str, Any]:
    results = []
    scores = []
    missing_predictions = 0
    errored = 0

    for idx, sample in enumerate(tqdm(gold, desc=f"Evaluating {model_name}")):
        qid = get_question_id(sample, idx)
        question = get_question(sample)
        reference_list = get_reference_list(sample)
        question_type = normalize_question_type(sample.get("question_type", "automatic"))
        candidate = pred_by_id.get(qid, "")

        if not candidate:
            missing_predictions += 1

        try:
            score = evaluation_utils.evaluate_example(
                question=question,
                reference_list=reference_list,
                candidate=candidate,
                question_type=question_type,
            )
            error = ""
        except Exception as e:
            score = 0.0
            error = str(e)
            errored += 1

        score = float(score)
        scores.append(score)

        record = records_by_id.get(qid, {})
        latency = safe_float(record.get("latency_seconds"))
        num_tool_calls = record.get("num_tool_calls")
        retrieved_urls = record.get("retrieved_urls", [])
        retrieval_recall_at_k = record.get("retrieval_recall_at_k")

        results.append({
            "id": qid,
            "question": question,
            "question_type": question_type,
            "reference_list": reference_list,
            "prediction": candidate,
            "score": score,
            "error": error,
            "latency_seconds": latency,
            "num_tool_calls": num_tool_calls,
            "retrieved_urls": retrieved_urls,
            "retrieval_recall_at_k": retrieval_recall_at_k,
        })

    by_type = {}

    for r in results:
        qt = r["question_type"]
        by_type.setdefault(qt, [])
        by_type[qt].append(r["score"])

    by_type_summary = {
        qt: {
            "count": len(vals),
            "score": mean(vals),
        }
        for qt, vals in by_type.items()
    }

    latencies = [
        r["latency_seconds"]
        for r in results
        if isinstance(r.get("latency_seconds"), float)
    ]

    return {
        "model": model_name,
        "summary": {
            "num_examples": len(results),
            "score": mean(scores),
            "missing_predictions": missing_predictions,
            "errored": errored,
            "by_question_type": by_type_summary,
            "avg_latency_seconds": mean(latencies) if latencies else None,
        },
        "results": results,
    }


def build_per_question_comparison(
    gold: List[Dict[str, Any]],
    model_outputs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    model_names = list(model_outputs.keys())

    results_by_model_and_id = {}

    for model_name, obj in model_outputs.items():
        results_by_model_and_id[model_name] = {
            r["id"]: r for r in obj["results"]
        }

    for idx, sample in enumerate(gold):
        qid = get_question_id(sample, idx)
        question = get_question(sample)
        question_type = normalize_question_type(sample.get("question_type", "automatic"))
        reference_list = get_reference_list(sample)

        row = {
            "id": qid,
            "question": question,
            "question_type": question_type,
            "reference_list": reference_list,
        }

        scores = {}

        for model_name in model_names:
            r = results_by_model_and_id[model_name].get(qid, {})
            row[f"{model_name}_prediction"] = r.get("prediction", "")
            row[f"{model_name}_score"] = r.get("score", 0.0)
            row[f"{model_name}_latency_seconds"] = r.get("latency_seconds", None)
            scores[model_name] = float(r.get("score", 0.0))

        best_score = max(scores.values()) if scores else 0.0
        winners = [m for m, s in scores.items() if s == best_score]

        row["best_score"] = best_score
        row["winner"] = "tie:" + ",".join(winners) if len(winners) > 1 else winners[0]

        rows.append(row)

    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        writer.writeheader()
        for row in rows:
            flat = {}
            for k, v in row.items():
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(v, ensure_ascii=False)
                else:
                    flat[k] = v
            writer.writerow(flat)


def write_markdown_report(
    path: Path,
    model_outputs: Dict[str, Dict[str, Any]],
    per_question_rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# EVQA model comparison report\n")
    lines.append("## Overall summary\n")
    lines.append("| Model | Score | Examples | Missing predictions | Errors | Avg latency (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for model_name, obj in model_outputs.items():
        s = obj["summary"]
        latency = s.get("avg_latency_seconds")
        latency_str = f"{latency:.4f}" if isinstance(latency, float) else "N/A"

        lines.append(
            f"| {model_name} | {s['score']:.4f} | {s['num_examples']} | "
            f"{s['missing_predictions']} | {s['errored']} | {latency_str} |"
        )

    lines.append("\n## Score by question type\n")

    all_qtypes = set()
    for obj in model_outputs.values():
        all_qtypes.update(obj["summary"]["by_question_type"].keys())

    model_names = list(model_outputs.keys())

    lines.append("| Question type | " + " | ".join(model_names) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(model_names)) + "|")

    for qt in sorted(all_qtypes):
        vals = []
        for model_name in model_names:
            qt_info = model_outputs[model_name]["summary"]["by_question_type"].get(qt)
            if qt_info:
                vals.append(f"{qt_info['score']:.4f} ({qt_info['count']})")
            else:
                vals.append("N/A")
        lines.append(f"| {qt} | " + " | ".join(vals) + " |")

    win_counts = {}

    for row in per_question_rows:
        winner = row["winner"]
        win_counts[winner] = win_counts.get(winner, 0) + 1

    lines.append("\n## Per-question winners\n")
    lines.append("| Winner | Count |")
    lines.append("|---|---:|")

    for winner, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {winner} | {count} |")

    lines.append("\n## Notes\n")
    lines.append("- `score` is the official EVQA score computed with Exact Match and BEM fallback.")
    lines.append("- `predictions_*.jsonl` are the files used for official scoring.")
    lines.append("- `records_*.jsonl` are optional and only used here for latency/retrieval/debug fields.")
    lines.append("- If BEM cannot be loaded, many examples may end up in `errored`.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gold", required=True)

    parser.add_argument("--vlm-pred", required=True)
    parser.add_argument("--rag-pred", required=True)
    parser.add_argument("--agentic-pred", required=True)

    parser.add_argument("--vlm-records", default=None)
    parser.add_argument("--rag-records", default=None)
    parser.add_argument("--agentic-records", default=None)

    parser.add_argument(
        "--eval-utils-dir",
        default="external/encyclopedic_vqa",
        help="Cartella che contiene evaluation_utils.py ufficiale."
    )

    parser.add_argument(
        "--out-dir",
        default="comparison_report",
        help="Cartella dove salvare report finale."
    )

    args = parser.parse_args()

    eval_utils_dir = Path(args.eval_utils_dir).resolve()
    sys.path.insert(0, str(eval_utils_dir))

    import evaluation_utils  # noqa: E402

    gold = load_json_or_jsonl(Path(args.gold))

    model_specs = {
        "baseline_vlm": {
            "pred": Path(args.vlm_pred),
            "records": Path(args.vlm_records) if args.vlm_records else None,
        },
        "standard_rag": {
            "pred": Path(args.rag_pred),
            "records": Path(args.rag_records) if args.rag_records else None,
        },
        "agentic_rag": {
            "pred": Path(args.agentic_pred),
            "records": Path(args.agentic_records) if args.agentic_records else None,
        },
    }

    model_outputs = {}

    for model_name, spec in model_specs.items():
        pred_by_id = load_predictions(spec["pred"])
        records_by_id = load_records_optional(spec["records"])

        model_outputs[model_name] = evaluate_model_on_gold(
            model_name=model_name,
            gold=gold,
            pred_by_id=pred_by_id,
            records_by_id=records_by_id,
            evaluation_utils=evaluation_utils,
        )

    per_question_rows = build_per_question_comparison(gold, model_outputs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        model_name: obj["summary"]
        for model_name, obj in model_outputs.items()
    }

    write_json(out_dir / "comparison_summary.json", summary)
    write_json(out_dir / "comparison_full_results.json", model_outputs)
    write_jsonl(out_dir / "comparison_per_question.jsonl", per_question_rows)
    write_csv(out_dir / "comparison_table.csv", per_question_rows)
    write_markdown_report(out_dir / "comparison_report.md", model_outputs, per_question_rows)

    print("\n=== FINAL COMPARISON SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\nReport salvato in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()