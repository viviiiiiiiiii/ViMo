import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


def normalize_question_type(qtype: str) -> str:
    """
    Lo script ufficiale accetta:
        templated, automatic, multi_answer, 2_hop

    Ma nel dataset possono comparire:
        multi-answer, 2-hop

    Quindi normalizziamo.
    """
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


def load_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_reference_list(sample):
    """
    Il dataset può avere:
    - answers: lista già pronta
    - answer: stringa unica, a volte con separatori | o &&
    """
    if "answers" in sample and sample["answers"]:
        return sample["answers"]

    answer = str(sample.get("answer", "")).strip()

    if not answer:
        return []

    refs = []
    for part in answer.split("|"):
        part = part.strip()
        if part and part not in refs:
            refs.append(part)

    return refs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gold", required=True, help="JSON domande EVQA con answer/answers.")
    parser.add_argument("--pred", required=True, help="JSON/JSONL predizioni con id + prediction.")
    parser.add_argument("--out", default="evqa_scores.json", help="Output dettagliato.")
    parser.add_argument(
        "--eval-utils-dir",
        default="external/encyclopedic_vqa",
        help="Cartella che contiene evaluation_utils.py ufficiale.",
    )

    args = parser.parse_args()

    eval_utils_dir = Path(args.eval_utils_dir).resolve()
    sys.path.insert(0, str(eval_utils_dir))

    import evaluation_utils  # noqa: E402

    gold = load_json_or_jsonl(Path(args.gold))
    pred = load_json_or_jsonl(Path(args.pred))

    pred_by_id = {}
    for p in pred:
        pred_by_id[str(p["id"])] = str(p.get("prediction", ""))

    results = []
    scores = []

    missing_predictions = 0
    errored = 0

    for sample in tqdm(gold, desc="Evaluating"):
        sample_id = str(sample["id"])
        question = str(sample.get("question", ""))
        candidate = pred_by_id.get(sample_id, "")

        if not candidate:
            missing_predictions += 1

        reference_list = get_reference_list(sample)
        question_type = normalize_question_type(sample.get("question_type", ""))

        try:
            score = evaluation_utils.evaluate_example(
                question=question,
                reference_list=reference_list,
                candidate=candidate,
                question_type=question_type,
            )
        except Exception as e:
            score = 0.0
            errored += 1
            error_msg = str(e)
        else:
            error_msg = ""

        result = {
            "id": sample_id,
            "question": question,
            "question_type": question_type,
            "reference_list": reference_list,
            "candidate": candidate,
            "score": float(score),
            "error": error_msg,
        }

        results.append(result)
        scores.append(float(score))

    avg_score = sum(scores) / len(scores) if scores else 0.0

    by_type = {}
    for r in results:
        qt = r["question_type"]
        by_type.setdefault(qt, [])
        by_type[qt].append(r["score"])

    by_type_scores = {
        qt: {
            "count": len(vals),
            "score": sum(vals) / len(vals) if vals else 0.0,
        }
        for qt, vals in by_type.items()
    }

    summary = {
        "num_examples": len(results),
        "score": avg_score,
        "missing_predictions": missing_predictions,
        "errored": errored,
        "by_question_type": by_type_scores,
    }

    output = {
        "summary": summary,
        "results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()