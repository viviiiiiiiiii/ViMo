"""
Utility condivise per salvare metadati di valutazione.
Non dipende da pandas/scipy e può essere usato direttamente sul server.
"""

import csv
import json
import math
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple


URL_RE = re.compile(r"\[URL_DOC:\s*([^\]]+)\]|URL_DOC:\s*([^\s\]\)]+)")
SECTION_RE = re.compile(r"Sezione\s+(\d+)\s*:\s*([^\n]+)")


def now_seconds() -> float:
    return time.perf_counter()


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def normalize_text(text: Any) -> str:
    text = safe_str(text).lower().strip()
    text = re.sub(r"[^\w\sàèéìòùç]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_estimate(text: Any) -> int:
    """Stima grezza ma stabile: utile se il modello non espone usage tokens."""
    text = safe_str(text)
    if not text:
        return 0
    # approssimazione robusta: parole + punteggiatura significativa
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def contains_match(answer: Any, ground_truth: Any) -> int:
    a = normalize_text(answer)
    gt = normalize_text(ground_truth)
    if not a or not gt:
        return 0
    return int(gt in a or a in gt)


def exact_match(answer: Any, ground_truth: Any) -> int:
    return int(normalize_text(answer) == normalize_text(ground_truth) and normalize_text(ground_truth) != "")


def semantic_similarity(answer: Any, ground_truth: Any) -> float:
    """
    Similarità lessicale/semantica leggera, senza caricare modelli aggiuntivi.
    Non sostituisce BEM/LLM-judge, ma dà una metrica automatica immediata.
    """
    a = normalize_text(answer)
    gt = normalize_text(ground_truth)
    if not a or not gt:
        return 0.0
    seq = SequenceMatcher(None, a, gt).ratio()
    aset, gtset = set(a.split()), set(gt.split())
    if not aset or not gtset:
        jacc = 0.0
        recall = 0.0
    else:
        jacc = len(aset & gtset) / len(aset | gtset)
        recall = len(aset & gtset) / len(gtset)
    return round(0.45 * seq + 0.35 * recall + 0.20 * jacc, 4)


def automatic_correct(answer: Any, ground_truth: Any, threshold: float = 0.72) -> int:
    """Correttezza automatica prudente: exact OR contains OR similarità sopra soglia."""
    if exact_match(answer, ground_truth):
        return 1
    if contains_match(answer, ground_truth):
        return 1
    return int(semantic_similarity(answer, ground_truth) >= threshold)


def unsupported_or_error(answer: Any) -> int:
    a = safe_str(answer).lower()
    markers = ["errore", "error", "non lo so", "non posso determinar", "not enough information"]
    return int(any(m in a for m in markers))


def hallucination_proxy(answer: Any, ground_truth: Any) -> int:
    """
    Proxy semplice: segnala come possibile allucinazione risposte non corrette,
    non vuote e non esplicitamente incerte/errore.
    Da rifinire manualmente se serve.
    """
    a = normalize_text(answer)
    if not a:
        return 0
    if automatic_correct(answer, ground_truth):
        return 0
    if unsupported_or_error(answer):
        return 0
    return 1


def parse_retrieved_urls(context: Any) -> List[str]:
    text = safe_str(context)
    urls = []
    for m in URL_RE.finditer(text):
        url = m.group(1) or m.group(2)
        if url and url not in urls:
            urls.append(url.strip())
    return urls


def parse_sections(context: Any) -> List[Dict[str, str]]:
    text = safe_str(context)
    return [{"section_idx": m.group(1), "section_title": m.group(2).strip()} for m in SECTION_RE.finditer(text)]


def retrieval_metrics(retrieved: Iterable[str], expected: Optional[Iterable[str]] = None, k: Optional[int] = None) -> Dict[str, Any]:
    retrieved_list = [safe_str(x) for x in (retrieved or []) if safe_str(x)]
    if k is not None:
        retrieved_list = retrieved_list[:k]
    expected_list = [safe_str(x) for x in (expected or []) if safe_str(x)]
    if not expected_list:
        return {
            "retrieval_recall_at_k": None,
            "retrieval_precision_at_k": None,
            "retrieval_mrr": None,
            "retrieved_expected_overlap": None,
        }

    def match(ret: str, exp: str) -> bool:
        rn, en = normalize_text(ret), normalize_text(exp)
        return bool(rn and en and (rn in en or en in rn))

    relevant_positions = []
    relevant_count = 0
    for i, r in enumerate(retrieved_list, start=1):
        if any(match(r, e) for e in expected_list):
            relevant_count += 1
            relevant_positions.append(i)

    recall = int(relevant_count > 0)
    precision = relevant_count / max(len(retrieved_list), 1)
    mrr = 1 / relevant_positions[0] if relevant_positions else 0.0
    return {
        "retrieval_recall_at_k": recall,
        "retrieval_precision_at_k": round(precision, 4),
        "retrieval_mrr": round(mrr, 4),
        "retrieved_expected_overlap": relevant_count,
    }


def build_common_record(
    *,
    question_id: Any,
    model_name: str,
    image_path: str,
    question: str,
    ground_truth: str,
    answer: str,
    question_type: Optional[str] = None,
    latency_seconds: Optional[float] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_tokens = token_estimate(question)
    output_tokens = token_estimate(answer)
    gt = safe_str(ground_truth)
    record = {
        "question_id": question_id,
        "model": model_name,
        "image": image_path,
        "question": question,
        "question_type": question_type or "unknown",
        "ground_truth": gt,
        "answer": answer,
        "exact_match": exact_match(answer, gt),
        "contains_match": contains_match(answer, gt),
        "semantic_score": semantic_similarity(answer, gt),
        "correct_auto": automatic_correct(answer, gt),
        "faithfulness_score": None,  # opzionale: puoi riempirlo con LLM judge/manuale
        "hallucination_proxy": hallucination_proxy(answer, gt),
        "latency_seconds": latency_seconds,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "total_tokens_est": input_tokens + output_tokens,
        "error": error,
    }
    if extra:
        record.update(extra)
    return record


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def flatten_value(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return v


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: flatten_value(r.get(k)) for k in keys})


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
