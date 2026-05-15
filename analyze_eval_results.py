import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from eval_utils import read_jsonl, write_csv

NUMERIC_FIELDS = [
    "correct_auto",
    "exact_match",
    "contains_match",
    "semantic_score",
    "faithfulness_score",
    "hallucination_proxy",
    "latency_seconds",
    "input_tokens_est",
    "output_tokens_est",
    "total_tokens_est",
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "retrieval_mrr",
    "num_steps",
    "num_tool_calls",
    "num_retrieval_calls",
    "num_section_reads",
]


def to_float(x):
    if x is None or x == "":
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def mean(values):
    vals = [to_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def group_by(rows: List[Dict], keys: Tuple[str, ...]):
    d = defaultdict(list)
    for r in rows:
        d[tuple(r.get(k, "unknown") for k in keys)].append(r)
    return d


def aggregate(rows: List[Dict], group_keys=("model",)) -> List[Dict]:
    output = []
    for key, group in group_by(rows, group_keys).items():
        rec = {group_keys[i]: key[i] for i in range(len(group_keys))}
        rec["n"] = len(group)
        for field in NUMERIC_FIELDS:
            rec[f"avg_{field}"] = mean([r.get(field) for r in group])
        # Metriche leggibili
        rec["accuracy"] = rec.pop("avg_correct_auto")
        rec["exact_match_rate"] = rec.pop("avg_exact_match")
        rec["contains_match_rate"] = rec.pop("avg_contains_match")
        rec["hallucination_rate_proxy"] = rec.pop("avg_hallucination_proxy")
        output.append(rec)
    return sorted(output, key=lambda x: tuple(str(x.get(k, "")) for k in group_keys))


def paired_comparisons(rows: List[Dict]) -> List[Dict]:
    """
    Confronta i modelli sulle stesse question_id.
    Produce delta accuracy, win/loss/tie e McNemar approssimato senza scipy.
    """
    by_q = defaultdict(dict)
    for r in rows:
        by_q[r.get("question_id")][r.get("model")] = r

    models = sorted({r.get("model") for r in rows})
    comps = []

    def norm_correct(r):
        return int(to_float(r.get("correct_auto")) or 0)

    for i, a in enumerate(models):
        for b in models[i+1:]:
            paired = [(m[a], m[b]) for m in by_q.values() if a in m and b in m]
            if not paired:
                continue
            a_correct = sum(norm_correct(x) for x, _ in paired)
            b_correct = sum(norm_correct(y) for _, y in paired)
            n = len(paired)

            a_wins = 0  # a corretto, b sbagliato
            b_wins = 0  # b corretto, a sbagliato
            ties = 0
            for x, y in paired:
                xc, yc = norm_correct(x), norm_correct(y)
                if xc > yc:
                    a_wins += 1
                elif yc > xc:
                    b_wins += 1
                else:
                    ties += 1

            acc_a = a_correct / n
            acc_b = b_correct / n
            delta = acc_a - acc_b
            err_a = 1 - acc_a
            err_b = 1 - acc_b
            error_reduction_a_vs_b = ((err_b - err_a) / err_b) if err_b > 0 else None

            # McNemar con correzione di continuità: chi2 = (|b-c|-1)^2/(b+c)
            discordant = a_wins + b_wins
            if discordant > 0:
                chi2 = (abs(a_wins - b_wins) - 1) ** 2 / discordant
                # p-value approssimato per chi-quadro 1 dof: erfc(sqrt(x/2))
                p_approx = math.erfc(math.sqrt(max(chi2, 0) / 2))
            else:
                chi2 = 0.0
                p_approx = 1.0

            comps.append({
                "model_a": a,
                "model_b": b,
                "n_paired": n,
                "accuracy_a": round(acc_a, 4),
                "accuracy_b": round(acc_b, 4),
                "delta_accuracy_a_minus_b": round(delta, 4),
                "relative_improvement_a_vs_b": round(delta / acc_b, 4) if acc_b > 0 else None,
                "error_reduction_a_vs_b": round(error_reduction_a_vs_b, 4) if error_reduction_a_vs_b is not None else None,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "mcnemar_chi2_approx": round(chi2, 4),
                "mcnemar_p_approx": round(p_approx, 4),
                "significant_0_05_approx": int(p_approx < 0.05),
            })
    return comps


def efficiency(rows: List[Dict]) -> List[Dict]:
    out = []
    for model, group in group_by(rows, ("model",)).items():
        model = model[0]
        correct = sum(int(to_float(r.get("correct_auto")) or 0) for r in group)
        tokens = sum(to_float(r.get("total_tokens_est")) or 0 for r in group)
        latency = sum(to_float(r.get("latency_seconds")) or 0 for r in group)
        out.append({
            "model": model,
            "n": len(group),
            "correct": correct,
            "total_tokens_est": round(tokens, 2),
            "total_latency_seconds": round(latency, 4),
            "correct_per_100k_tokens": round(correct / tokens * 100000, 4) if tokens > 0 else None,
            "correct_per_minute": round(correct / latency * 60, 4) if latency > 0 else None,
            "avg_tokens_per_correct": round(tokens / correct, 4) if correct > 0 else None,
            "avg_seconds_per_correct": round(latency / correct, 4) if correct > 0 else None,
        })
    return out


def write_markdown(path: str, overall, by_type, comps, eff):
    lines = []
    lines.append("# Report valutazione modelli\n")
    lines.append("## Performance generale\n")
    lines.extend(markdown_table(overall))
    lines.append("\n## Performance per categoria\n")
    lines.extend(markdown_table(by_type))
    lines.append("\n## Confronti paired\n")
    lines.extend(markdown_table(comps))
    lines.append("\n## Efficienza\n")
    lines.extend(markdown_table(eff))
    lines.append("\n## Lettura rapida\n")
    if overall:
        best_acc = max(overall, key=lambda r: r.get("accuracy") or -1)
        fastest = min(overall, key=lambda r: r.get("avg_latency_seconds") if r.get("avg_latency_seconds") is not None else 10**9)
        lines.append(f"- Migliore accuracy automatica: **{best_acc['model']}** ({best_acc.get('accuracy')}).")
        lines.append(f"- Minore latenza media: **{fastest['model']}** ({fastest.get('avg_latency_seconds')} s).")
    if eff:
        best_eff = max(eff, key=lambda r: r.get("correct_per_100k_tokens") or -1)
        lines.append(f"- Migliore efficienza per token: **{best_eff['model']}** ({best_eff.get('correct_per_100k_tokens')} corrette / 100k token stimati).")
    lines.append("\nNota: `correct_auto`, `hallucination_proxy` e `semantic_score` sono metriche automatiche leggere. Per risultati da tesi, conviene aggiungere anche una colonna manuale o LLM-judge.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def markdown_table(rows: List[Dict]) -> List[str]:
    if not rows:
        return ["_Nessun dato._\n"]
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    lines = []
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Analizza eval_results.jsonl e produce metriche definitive/confronti.")
    parser.add_argument("--input", default="eval_results.jsonl", help="File JSONL prodotto da evaluate_all.py")
    parser.add_argument("--out-dir", default="eval_report", help="Cartella output report")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_jsonl(args.input)

    overall = aggregate(rows, ("model",))
    by_type = aggregate(rows, ("question_type", "model"))
    comps = paired_comparisons(rows)
    eff = efficiency(rows)

    write_csv(os.path.join(args.out_dir, "metrics_overall.csv"), overall)
    write_csv(os.path.join(args.out_dir, "metrics_by_question_type.csv"), by_type)
    write_csv(os.path.join(args.out_dir, "paired_comparisons.csv"), comps)
    write_csv(os.path.join(args.out_dir, "efficiency.csv"), eff)

    summary = {
        "overall": overall,
        "by_question_type": by_type,
        "paired_comparisons": comps,
        "efficiency": eff,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_markdown(os.path.join(args.out_dir, "report.md"), overall, by_type, comps, eff)

    print(f"✅ Analisi completata. Report in: {args.out_dir}")
    print(f"- {os.path.join(args.out_dir, 'report.md')}")
    print(f"- {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
