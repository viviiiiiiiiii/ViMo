#!/usr/bin/env python3

import argparse
import gzip
import json
import os
import random
import shutil
import tarfile
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm


INAT_VAL_JSON_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz"
INAT_BASE_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"


def download_file(url: str, dst: Path, retries: int = 3) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        return True

    tmp = dst.with_suffix(dst.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    print(f"[WARN] HTTP {r.status_code}: {url}")
                    time.sleep(1)
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
            print(f"[WARN] Tentativo {attempt}/{retries} fallito per {url}: {e}")
            time.sleep(2)

    if tmp.exists():
        tmp.unlink()

    return False


def download_annotation_json(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    tar_path = cache_dir / "val.json.tar.gz"
    json_path = cache_dir / "val.json"

    if json_path.exists():
        print(f"[OK] Annotazioni già presenti: {json_path}")
        return json_path

    ok = download_file(INAT_VAL_JSON_URL, tar_path)
    if not ok:
        raise RuntimeError("Non sono riuscito a scaricare val.json.tar.gz")

    print("[EXTRACT] Estraggo val.json...")

    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".json")]
        if not members:
            raise RuntimeError("Nessun JSON trovato dentro val.json.tar.gz")

        member = members[0]
        with tar.extractfile(member) as src, open(json_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    return json_path


def load_inat_mapping(annotation_json: Path) -> dict:
    print(f"[LOAD] Carico mappa id -> file_name da {annotation_json}")

    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_file = {}

    for img in data["images"]:
        id_to_file[str(img["id"])] = img["file_name"].lstrip("./")

    print(f"[OK] Immagini indicizzate: {len(id_to_file)}")
    return id_to_file


def split_field(value):
    if value is None:
        return []

    if pd.isna(value):
        return []

    return [x.strip() for x in str(value).split("|") if x.strip()]


def split_answers(answer):
    answers = []

    for part in split_field(answer):
        for sub in part.split("&&"):
            sub = sub.strip()
            if sub and sub not in answers:
                answers.append(sub)

    return answers


def make_direct_image_url(file_name: str) -> str:
    """
    file_name esempio:
        val/00000_Animalia_Arthropoda_Insecta/.../image.jpg

    Costruiamo:
        https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val/...
    """
    clean = file_name.lstrip("/")
    return f"{INAT_BASE_URL}/{quote(clean)}"


def download_one_image(file_name: str, out_dir: Path) -> tuple[bool, str, str]:
    """
    Ritorna:
        ok, relative_path, url
    """

    local_path = out_dir / "images" / file_name
    rel_path = str(Path("images") / file_name)

    if local_path.exists() and local_path.stat().st_size > 0:
        return True, rel_path, make_direct_image_url(file_name)

    url = make_direct_image_url(file_name)
    ok = download_file(url, local_path)

    return ok, rel_path, url


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, default="data/evqa_1000_direct")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--images-per-question", type=str, default="1", choices=["1", "all"])

    args = parser.parse_args()

    out_dir = Path(args.out)
    cache_dir = out_dir / "_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[LOAD] Carico Encyclopedic-VQA da Hugging Face...")
    ds = load_dataset("reonokiy/vsp-encyclopedic-vqa", split=args.split)
    df = pd.DataFrame(ds)

    df = df[df["dataset_name"] == "inaturalist"].copy()

    if len(df) < args.n:
        raise RuntimeError(
            f"Ho trovato solo {len(df)} esempi iNaturalist nello split {args.split}, "
            f"ma ne hai chiesti {args.n}."
        )

    df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    annotation_json = download_annotation_json(cache_dir)
    id_to_file = load_inat_mapping(annotation_json)

    records = []
    all_required_files = set()

    print("[BUILD] Costruisco record domande + file immagini...")

    for i, row in df.iterrows():
        image_ids = split_field(row.get("dataset_image_ids"))

        if args.images_per_question == "1":
            image_ids = image_ids[:1]

        image_file_names = []
        missing_image_ids = []

        for image_id in image_ids:
            file_name = id_to_file.get(str(image_id))

            if file_name is None:
                missing_image_ids.append(image_id)
                continue

            image_file_names.append(file_name)
            all_required_files.add(file_name)

        record = {
            "id": f"evqa_{args.split}_{i:05d}",
            "source_dataset": "encyclopedic_vqa",
            "split": args.split,
            "dataset_name": row.get("dataset_name"),

            "question": row.get("question"),
            "question_original": row.get("question_original"),
            "question_type": row.get("question_type"),

            "answer": row.get("answer"),
            "answers": split_answers(row.get("answer")),

            "wikipedia_title": row.get("wikipedia_title"),
            "wikipedia_url": row.get("wikipedia_url"),

            "evidence": row.get("evidence"),
            "evidence_section_id": row.get("evidence_section_id"),
            "evidence_section_title": row.get("evidence_section_title"),

            "dataset_category_id": row.get("dataset_category_id"),
            "dataset_image_ids": image_ids,

            "image_file_names": image_file_names,
            "image_paths": [],
            "image_urls": [],

            "missing_image_ids": missing_image_ids,
            "download_ok": False,
        }

        records.append(record)

    print(f"[INFO] Domande selezionate: {len(records)}")
    print(f"[INFO] Immagini uniche da scaricare: {len(all_required_files)}")

    downloaded = {}
    failed = {}

    print("[DOWNLOAD] Scarico solo le immagini necessarie...")

    for file_name in tqdm(sorted(all_required_files), desc="Images"):
        ok, rel_path, url = download_one_image(file_name, out_dir)

        if ok:
            downloaded[file_name] = {
                "path": rel_path,
                "url": url,
            }
        else:
            failed[file_name] = url

    for record in records:
        paths = []
        urls = []
        ok_flags = []

        for file_name in record["image_file_names"]:
            item = downloaded.get(file_name)

            if item:
                paths.append(item["path"])
                urls.append(item["url"])
                ok_flags.append(True)
            else:
                paths.append(str(Path("images") / file_name))
                urls.append(make_direct_image_url(file_name))
                ok_flags.append(False)

        record["image_paths"] = paths
        record["image_urls"] = urls
        record["download_ok"] = bool(ok_flags) and all(ok_flags)

    out_json = out_dir / f"evqa_{args.split}_{args.n}.json"
    failed_json = out_dir / "failed_images.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    with open(failed_json, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    print()
    print("[DONE]")
    print(f"JSON finale: {out_json}")
    print(f"Immagini scaricate: {len(downloaded)} / {len(all_required_files)}")
    print(f"Immagini fallite: {len(failed)}")

    if failed:
        print()
        print("[WARN] Alcune immagini non sono state scaricate.")
        print(f"Lista fallimenti salvata in: {failed_json}")
        print("In quel caso puoi usare la versione fallback che estrae dal tar completo.")

    print()
    print("Esempio record:")
    print(json.dumps(records[0], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
    
    
    