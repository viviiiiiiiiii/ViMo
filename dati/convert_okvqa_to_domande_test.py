from pathlib import Path
import json
import shutil
from collections import Counter

# =========================
# CONFIGURAZIONE
# =========================

ROOT = Path(__file__).resolve().parent.parent

RAW_OKVQA_DIR = ROOT / "dati_raw" / "okvqa"

# Usa val2014 per il primo test.
QUESTIONS_PATH = RAW_OKVQA_DIR / "OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_PATH = RAW_OKVQA_DIR / "mscoco_val2014_annotations.json"
COCO_IMAGES_DIR = RAW_OKVQA_DIR / "val2014"

OUT_IMAGES_DIR = ROOT / "dati" / "immagini"
OUT_JSON_PATH = ROOT / "dati" / "domande_test.json"

# Numero massimo di domande da convertire.
# Parti basso, poi aumenta.
MAX_SAMPLES = 500

# Se True copia fisicamente le immagini in dati/immagini.
# Se False mette solo il path alla cartella originale.
COPY_IMAGES = True

# Prefisso da mettere in related_images dentro domande_test.json.
# Con COPY_IMAGES=True, le immagini stanno in dati/immagini.
RELATED_IMAGE_PREFIX = "dati/immagini"


# =========================
# FUNZIONI
# =========================

def coco_filename(image_id: int, split: str = "val2014") -> str:
    """
    Converte image_id COCO nel nome file standard.
    Esempio:
    image_id=42 -> COCO_val2014_000000000042.jpg
    """
    return f"COCO_{split}_{image_id:012d}.jpg"


def normalize_answer(ans: str) -> str:
    """
    Pulizia minima delle risposte.
    Non facciamo stemming o normalizzazioni aggressive,
    perché per ora ci serve solo salvare ground truth leggibili.
    """
    if ans is None:
        return ""
    return " ".join(str(ans).strip().lower().split())


def extract_unique_answers(annotation: dict):
    """
    Estrae le risposte da OK-VQA.
    Mantiene risposte uniche ordinate per frequenza decrescente.
    """
    raw_answers = []

    for a in annotation.get("answers", []):
        ans = normalize_answer(a.get("answer", ""))
        if ans:
            raw_answers.append(ans)

    if not raw_answers and annotation.get("multiple_choice_answer"):
        raw_answers.append(normalize_answer(annotation["multiple_choice_answer"]))

    counts = Counter(raw_answers)

    # Risposte uniche ordinate: prima quelle più frequenti.
    unique_answers = [
        ans for ans, _ in counts.most_common()
        if ans
    ]

    return unique_answers


def main():
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Leggo domande da: {QUESTIONS_PATH}")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    print(f"Leggo annotazioni da: {ANNOTATIONS_PATH}")
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)

    questions = questions_data["questions"]
    annotations = annotations_data["annotations"]

    # Mappa question_id -> annotazione
    ann_by_qid = {
        ann["question_id"]: ann
        for ann in annotations
    }

    converted = []
    skipped_missing_ann = 0
    skipped_missing_img = 0
    copied_images = 0

    for q in questions:
        if len(converted) >= MAX_SAMPLES:
            break

        qid = q["question_id"]
        image_id = q["image_id"]
        question_text = q["question"].strip()

        ann = ann_by_qid.get(qid)
        if ann is None:
            skipped_missing_ann += 1
            continue

        answers = extract_unique_answers(ann)
        if not answers:
            continue

        filename = coco_filename(image_id, split="val2014")
        src_img = COCO_IMAGES_DIR / filename

        if not src_img.exists():
            skipped_missing_img += 1
            continue

        if COPY_IMAGES:
            dst_img = OUT_IMAGES_DIR / filename

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                copied_images += 1

            related_image_path = f"{RELATED_IMAGE_PREFIX}/{filename}"
        else:
            # In questo caso il path punta ai dati raw.
            related_image_path = str(src_img)

        converted.append({
            "question_id": qid,
            "image_id": image_id,
            "question": question_text,
            "related_images": related_image_path,
            "answer": answers,
            "question_type": "okvqa"
        })

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print("\nFATTO.")
    print(f"Domande convertite: {len(converted)}")
    print(f"Immagini copiate: {copied_images}")
    print(f"Annotazioni mancanti: {skipped_missing_ann}")
    print(f"Immagini mancanti: {skipped_missing_img}")
    print(f"Output JSON: {OUT_JSON_PATH}")
    print(f"Cartella immagini: {OUT_IMAGES_DIR}")


if __name__ == "__main__":
    main()