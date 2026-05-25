import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from baseline_rag import run_standard_rag


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    for key in ["data", "examples", "questions", "samples", "items"]:
        if key in obj and isinstance(obj[key], list):
            return obj[key]

    raise ValueError(f"Formato JSON non riconosciuto in {path}")


def get_first_existing(sample: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        value = sample.get(key)
        if value is not None and value != "":
            return value
    return default


def get_question_id(sample: Dict[str, Any], fallback_idx: int) -> str:
    return str(
        get_first_existing(
            sample,
            ["id", "unique_id", "question_id", "qid", "sample_id"],
            default=fallback_idx
        )
    )


def get_question(sample: Dict[str, Any]) -> str:
    return str(get_first_existing(sample, ["question", "query", "text"], default=""))


def get_image_path(sample: Dict[str, Any], image_root: Optional[Path]) -> str:
    raw_path = get_first_existing(
        sample,
        ["related_images", "image_path", "image", "image_file", "filename", "file_name", "img_path"],
        default=""
    )

    raw_path = str(raw_path)

    if not raw_path:
        return ""

    path = Path(raw_path)

    if path.is_absolute():
        return str(path)

    if image_root is not None:
        return str(image_root / path)

    return raw_path


def get_ground_truth(sample: Dict[str, Any]) -> str:
    if "answers" in sample and isinstance(sample["answers"], list):
        return " | ".join(str(a) for a in sample["answers"])

    return str(get_first_existing(sample, ["answer", "ground_truth", "gt"], default=""))


def get_expected_sources(sample: Dict[str, Any]) -> List[str]:
    sources = get_first_existing(
        sample,
        ["expected_sources", "sources", "source_urls", "gold_sources"],
        default=[]
    )

    if sources is None:
        return []

    if isinstance(sources, list):
        return [str(s) for s in sources]

    return [str(sources)]


def extract_answer_from_record(record: Dict[str, Any]) -> str:
    for key in ["answer", "prediction", "output", "response"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return ""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--pred-out", default="predictions_baseline.jsonl")
    parser.add_argument("--records-out", default="records_baseline.jsonl")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    pred_out_path = Path(args.pred_out)
    records_out_path = Path(args.records_out)
    image_root = Path(args.image_root) if args.image_root else None

    samples = load_json_or_jsonl(input_path)

    if args.limit is not None:
        samples = samples[:args.limit]

    already_done = set()

    if args.skip_existing and pred_out_path.exists():
        with open(pred_out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    already_done.add(str(obj["id"]))
                except Exception:
                    pass

    pred_out_path.parent.mkdir(parents=True, exist_ok=True)
    records_out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_mode = "a" if args.skip_existing else "w"
    records_mode = "a" if args.skip_existing else "w"

    with open(pred_out_path, pred_mode, encoding="utf-8") as pred_f, \
         open(records_out_path, records_mode, encoding="utf-8") as records_f:

        for idx, sample in enumerate(tqdm(samples, desc="Generating baseline predictions")):
            question_id = get_question_id(sample, idx)

            if question_id in already_done:
                continue

            question = get_question(sample)
            image_path = get_image_path(sample, image_root)
            if image_path and not Path(image_path).exists():
                print(f"⚠️ IMAGE NOT FOUND for id={question_id}: {image_path}")
            ground_truth = get_ground_truth(sample)
            question_type = str(sample.get("question_type", "unknown"))
            expected_sources = get_expected_sources(sample)

            if not question:
                record = {
                    "question_id": question_id,
                    "model_name": "standard_rag",
                    "image_path": image_path,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": "",
                    "question_type": question_type,
                    "latency_seconds": 0,
                    "error": "Missing question field",
                    "extra": {}
                }
            elif not image_path:
                record = {
                    "question_id": question_id,
                    "model_name": "standard_rag",
                    "image_path": image_path,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": "",
                    "question_type": question_type,
                    "latency_seconds": 0,
                    "error": "Missing image path field",
                    "extra": {}
                }
            else:
                record = run_standard_rag(
                    image_path=image_path,
                    question=question,
                    return_metadata=True,
                    question_id=question_id,
                    ground_truth=ground_truth,
                    question_type=question_type,
                    expected_sources=expected_sources
                )

            prediction = extract_answer_from_record(record)

            pred_row = {
                "id": question_id,
                "prediction": prediction
            }

            pred_f.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            pred_f.flush()

            records_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_f.flush()


if __name__ == "__main__":
    main()