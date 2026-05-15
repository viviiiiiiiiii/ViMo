#!/usr/bin/env python3

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm


# Base ufficiale GLDv2.
# Proviamo accesso diretto ai jpg nel path estratto.
# Se S3 non espone i singoli jpg, questi download possono fallire.
GLD_S3_BASE = "https://s3.amazonaws.com/google-landmark"


def is_nan_like(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return False


def clean_str(x) -> str:
    if x is None or is_nan_like(x):
        return ""
    return str(x).strip()


def split_pipe_field(value):
    value = clean_str(value)
    if not value:
        return []
    return [x.strip() for x in value.split("|") if x.strip()]


def split_answers(answer):
    answers = []
    for part in split_pipe_field(answer):
        for sub in part.split("&&"):
            sub = sub.strip()
            if sub and sub not in answers:
                answers.append(sub)
    return answers


def normalize_question_type(qtype: str) -> str:
    qtype = clean_str(qtype).lower()
    qtype = qtype.replace("_", "-").replace(" ", "-")
    return qtype


def is_single_hop(qtype: str) -> bool:
    qtype = normalize_question_type(qtype)

    # In Encyclopedic-VQA, nello split che stai usando, i tipi sono:
    # automatic, templated, multi-answer, 2-hop.
    # Quindi consideriamo single-hop tutte le domande che NON sono 2-hop.
    return qtype not in {"2-hop", "two-hop", "2hop", "twohop"}


def gld_relative_image_path(image_id: str, gld_split: str = "index") -> str:
    """
    GLDv2 ufficiale, una volta estratto, usa:
        index/a/b/c/image_id.jpg

    Esempio:
        image_id = b19f656d8dfc4f0c
        path = index/b/1/9/b19f656d8dfc4f0c.jpg
    """
    image_id = clean_str(image_id)

    if len(image_id) < 3:
        raise ValueError(f"Image id troppo corto: {image_id}")

    return f"{gld_split}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"


def local_image_path(out_dir: Path, image_id: str, gld_split: str = "index") -> Path:
    return out_dir / "images" / "GLDv2" / gld_relative_image_path(image_id, gld_split)


def json_image_path(image_id: str, gld_split: str = "index") -> str:
    return str(Path("images") / "GLDv2" / gld_relative_image_path(image_id, gld_split))


def direct_gld_url(image_id: str, gld_split: str = "index") -> str:
    """
    URL diretto tentato.

    Nota:
    La documentazione ufficiale parla soprattutto di TAR shard.
    Questo URL può funzionare oppure no, dipende da cosa è esposto su S3.
    """
    rel = gld_relative_image_path(image_id, gld_split)
    return f"{GLD_S3_BASE}/{quote(rel)}"


def download_file(url: str, dst: Path, retries: int = 3, sleep_sec: float = 1.0) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        return True

    tmp = dst.with_suffix(dst.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    print(f"[WARN] HTTP {r.status_code}: {url}")
                    time.sleep(sleep_sec)
                    continue

                total = int(r.headers.get("content-length", 0))

                with open(tmp, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dst.name,
                    leave=False,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            tmp.rename(dst)
            return True

        except Exception as e:
            print(f"[WARN] Tentativo {attempt}/{retries} fallito: {url}")
            print(f"       Errore: {e}")
            time.sleep(sleep_sec)

    if tmp.exists():
        tmp.unlink()

    return False


def copy_from_existing_gld_root(
    image_id: str,
    out_dir: Path,
    gld_root: Path,
    gld_split: str = "index",
) -> bool:
    """
    Se hai già GLDv2 estratto da qualche parte, ad esempio:

        /work/datasets/GLDv2/index/b/1/9/b19f656d8dfc4f0c.jpg

    puoi passare:

        --gld-root /work/datasets/GLDv2

    e lo script copia solo le immagini necessarie nella cartella di output.
    """
    src = gld_root / gld_relative_image_path(image_id, gld_split)
    dst = local_image_path(out_dir, image_id, gld_split)

    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists() or dst.stat().st_size == 0:
        shutil.copy2(src, dst)

    return True


def build_records(df: pd.DataFrame, args) -> tuple[list, set]:
    records = []
    required_image_ids = set()

    for i, row in df.iterrows():
        image_ids = split_pipe_field(row.get("dataset_image_ids"))

        if args.images_per_question == "1":
            image_ids = image_ids[:1]

        image_paths = []
        image_urls = []

        for image_id in image_ids:
            image_paths.append(json_image_path(image_id, args.gld_split))
            image_urls.append(direct_gld_url(image_id, args.gld_split))
            required_image_ids.add(image_id)

        record = {
            "id": f"evqa_landmarks_singlehop_{i:05d}",
            "source_dataset": "encyclopedic_vqa",
            "split": args.split,
            "dataset_name": clean_str(row.get("dataset_name")),
            "gld_split": args.gld_split,

            "question": clean_str(row.get("question")),
            "question_original": clean_str(row.get("question_original")),
            "question_type": clean_str(row.get("question_type")),

            "answer": clean_str(row.get("answer")),
            "answers": split_answers(row.get("answer")),

            "wikipedia_title": clean_str(row.get("wikipedia_title")),
            "wikipedia_url": clean_str(row.get("wikipedia_url")),

            "evidence": clean_str(row.get("evidence")),
            "evidence_section_id": clean_str(row.get("evidence_section_id")),
            "evidence_section_title": clean_str(row.get("evidence_section_title")),

            "dataset_category_id": clean_str(row.get("dataset_category_id")),
            "dataset_image_ids": image_ids,

            "image_paths": image_paths,
            "image_urls": image_urls,

            "download_ok": False,
            "downloaded_image_paths": [],
            "failed_image_ids": [],
        }

        records.append(record)

    return records, required_image_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, default="data/evqa_1000_landmarks_singlehop")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])

    parser.add_argument(
        "--images-per-question",
        type=str,
        default="1",
        choices=["1", "all"],
        help="Usa una immagine per domanda oppure tutte le immagini associate.",
    )

    parser.add_argument(
        "--gld-split",
        type=str,
        default="index",
        choices=["index", "train", "test"],
        help=(
            "Split GLDv2 da usare per costruire i path immagine. "
            "Per Encyclopedic-VQA landmarks di solito è index."
        ),
    )

    parser.add_argument(
        "--gld-root",
        type=str,
        default="",
        help=(
            "Opzionale: path a GLDv2 già estratto. "
            "Esempio: /work/datasets/GLDv2, contenente index/a/b/c/id.jpg"
        ),
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Non scarica immagini, crea solo JSON con path previsti.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gld_root = Path(args.gld_root) if args.gld_root else None

    print("[LOAD] Carico Encyclopedic-VQA da Hugging Face...")
    ds = load_dataset("reonokiy/vsp-encyclopedic-vqa", split=args.split)
    df = pd.DataFrame(ds)

    print(f"[INFO] Colonne disponibili: {list(df.columns)}")
    print(f"[INFO] Esempi totali split {args.split}: {len(df)}")

    if "dataset_name" not in df.columns:
        raise RuntimeError("Nel dataset manca la colonna dataset_name.")

    if "question_type" not in df.columns:
        raise RuntimeError("Nel dataset manca la colonna question_type.")

    df["dataset_name_norm"] = df["dataset_name"].apply(lambda x: clean_str(x).lower())
    df["question_type_norm"] = df["question_type"].apply(normalize_question_type)

    print("[INFO] Valori dataset_name:")
    print(df["dataset_name_norm"].value_counts(dropna=False).head(20))

    print("[INFO] Valori question_type:")
    print(df["question_type_norm"].value_counts(dropna=False).head(20))

    df = df[df["dataset_name_norm"] == "landmarks"].copy()
    df = df[df["question_type"].apply(is_single_hop)].copy()

    print(f"[INFO] Esempi landmarks + single-hop disponibili: {len(df)}")

    if len(df) < args.n:
        raise RuntimeError(
            f"Ho trovato solo {len(df)} esempi landmarks single-hop nello split {args.split}, "
            f"ma ne hai chiesti {args.n}. Prova con --split train oppure riduci --n."
        )

    df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    records, required_image_ids = build_records(df, args)

    print(f"[INFO] Domande selezionate: {len(records)}")
    print(f"[INFO] Immagini uniche richieste: {len(required_image_ids)}")

    downloaded = {}
    failed = {}

    if args.no_download:
        print("[SKIP] --no-download attivo: creo solo il JSON.")
    else:
        print("[DOWNLOAD] Scarico/copio immagini GLDv2 richieste...")

        for image_id in tqdm(sorted(required_image_ids), desc="Images"):
            ok = False

            # 1. Se esiste un GLDv2 locale già estratto, copia da lì.
            if gld_root is not None:
                ok = copy_from_existing_gld_root(
                    image_id=image_id,
                    out_dir=out_dir,
                    gld_root=gld_root,
                    gld_split=args.gld_split,
                )

            # 2. Altrimenti prova download diretto dal path S3.
            if not ok:
                dst = local_image_path(out_dir, image_id, args.gld_split)
                url = direct_gld_url(image_id, args.gld_split)
                ok = download_file(url, dst)

            rel_path = json_image_path(image_id, args.gld_split)

            if ok:
                downloaded[image_id] = rel_path
            else:
                failed[image_id] = {
                    "expected_path": rel_path,
                    "tried_url": direct_gld_url(image_id, args.gld_split),
                }

    for record in records:
        downloaded_paths = []
        failed_ids = []

        for image_id in record["dataset_image_ids"]:
            if image_id in downloaded:
                downloaded_paths.append(downloaded[image_id])
            else:
                failed_ids.append(image_id)

        record["downloaded_image_paths"] = downloaded_paths
        record["failed_image_ids"] = failed_ids
        record["download_ok"] = len(record["dataset_image_ids"]) > 0 and len(failed_ids) == 0

    out_json = out_dir / f"evqa_{args.split}_landmarks_singlehop_{args.n}.json"
    failed_json = out_dir / "failed_images.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open(failed_json, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    print()
    print("[DONE]")
    print(f"JSON finale: {out_json}")
    print(f"Domande salvate: {len(records)}")
    print(f"Immagini richieste: {len(required_image_ids)}")
    print(f"Immagini scaricate/copiate: {len(downloaded)}")
    print(f"Immagini fallite: {len(failed)}")
    print(f"Failed JSON: {failed_json}")

    if failed:
        print()
        print("[ATTENZIONE]")
        print("Alcune immagini GLDv2 non sono state scaricate direttamente.")
        print("Questo può succedere perché GLDv2 ufficialmente distribuisce le immagini in TAR shard.")
        print("Se hai GLDv2 già estratto, rilancia passando --gld-root /path/al/GLDv2.")

    print()
    print("Esempio record:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2)[:2500])


if __name__ == "__main__":
    main()