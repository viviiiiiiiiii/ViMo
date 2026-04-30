from pathlib import Path
import json
import re
from datasets import load_dataset
from PIL import Image

# =========================
# CONFIGURAZIONE
# =========================

OUT_DIR = Path(__file__).resolve().parent
KB_IMAGES_DIR = OUT_DIR / "kb_images"

OUT_KB_PATH = OUT_DIR / "encyclopedic_kb_wiki.json"
OUT_KNN_PATH = OUT_DIR / "knn.json"

# Numero di documenti finali da creare.
# Parti basso: 100 per debug, poi 1000/5000 se tutto funziona.
TARGET_DOCS = 100

# Lingua preferita per i testi Wikipedia.
# "en" è la scelta più sicura perché Qwen e molti benchmark lavorano bene in inglese.
PREFERRED_LANGUAGE = "en"

# Lunghezza minima del testo finale per evitare documenti inutili.
MIN_TEXT_CHARS = 80


# =========================
# FUNZIONI DI SUPPORTO
# =========================

def clean_text(x):
    """Pulisce stringhe o valori mancanti."""
    if x is None:
        return ""
    if isinstance(x, list):
        x = " ".join(str(v) for v in x if v is not None)
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def get_first_valid(values):
    """Prende il primo valore non vuoto da una lista o da un campo singolo."""
    if values is None:
        return ""
    if not isinstance(values, list):
        return clean_text(values)
    for v in values:
        v = clean_text(v)
        if v:
            return v
    return ""


def pick_language_index(wit_features, preferred_language="en"):
    """
    Sceglie l'indice della lingua preferita dentro wit_features.
    Se non trova l'inglese, prende il primo esempio disponibile.
    """
    languages = wit_features.get("language", [])
    if not isinstance(languages, list) or len(languages) == 0:
        return None

    for i, lang in enumerate(languages):
        if lang == preferred_language:
            return i

    return 0


def get_wit_field(wit_features, field_name, idx):
    """
    Estrae un campo da wit_features.
    Molti campi sono liste parallele: page_title[i], page_url[i], caption[i], ecc.
    """
    values = wit_features.get(field_name, "")
    if isinstance(values, list):
        if idx is not None and idx < len(values):
            return clean_text(values[idx])
        return get_first_valid(values)
    return clean_text(values)


def safe_save_image(image, out_path):
    """
    Salva l'immagine PIL in RGB.
    Restituisce True se riesce, False altrimenti.
    """
    try:
        if image is None:
            return False
        image = image.convert("RGB")
        image.save(out_path, format="JPEG", quality=90)
        return True
    except Exception as e:
        print(f"[WARN] Impossibile salvare immagine {out_path}: {e}")
        return False


def build_section_texts(example, wit_features, idx):
    """
    Costruisce le section_texts compatibili col vostro codice.
    Qui mettiamo caption, alt text, descrizione pagina e contesto sezione.
    """
    caption_reference = get_wit_field(
        wit_features,
        "caption_reference_description",
        idx
    )

    caption_title_ref = get_wit_field(
        wit_features,
        "caption_title_and_reference_description",
        idx
    )

    caption_alt = get_wit_field(
        wit_features,
        "caption_alt_text_description",
        idx
    )

    context_page = get_wit_field(
        wit_features,
        "context_page_description",
        idx
    )

    context_section = get_wit_field(
        wit_features,
        "context_section_description",
        idx
    )

    attribution = clean_text(example.get("caption_attribution_description", ""))

    section_texts = []

    if caption_reference:
        section_texts.append(f"Caption: {caption_reference}")

    if caption_title_ref and caption_title_ref != caption_reference:
        section_texts.append(f"Title and caption: {caption_title_ref}")

    if caption_alt and caption_alt not in caption_reference:
        section_texts.append(f"Alt text: {caption_alt}")

    if context_page:
        section_texts.append(f"Page description: {context_page}")

    if context_section:
        section_texts.append(f"Section context: {context_section}")

    if attribution:
        section_texts.append(f"Wikimedia description: {attribution}")

    # Rimuove duplicati mantenendo ordine
    deduped = []
    seen = set()
    for t in section_texts:
        key = t.lower()
        if key not in seen:
            deduped.append(t)
            seen.add(key)

    return deduped


def main():
    KB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Carico wikimedia/wit_base da Hugging Face in streaming...")
    ds = load_dataset("wikimedia/wit_base", split="train", streaming=True)

    kb = {}
    knn = []

    scanned = 0
    kept = 0

    for example in ds:
        scanned += 1

        if kept >= TARGET_DOCS:
            break

        wit_features = example.get("wit_features", {})
        if not isinstance(wit_features, dict): #print(f"[WARN] wit_features non è un dizionario valido per esempio {scanned}")
            continue

        idx = pick_language_index(wit_features, PREFERRED_LANGUAGE) #idx = pick_language_index(wit_features, PREFERRED_LANGUAGE), se voglio che prenda solo l'inglese, altrimenti idx = 0 per prendere il primo disponibile
        if idx is None:
            continue

        title = get_wit_field(wit_features, "page_title", idx)
        page_url = get_wit_field(wit_features, "page_url", idx)

        if not title:
            continue

        section_texts = build_section_texts(example, wit_features, idx)
        merged_text = " ".join(section_texts)

        if len(merged_text) < MIN_TEXT_CHARS:
            continue

        doc_id = f"doc_{kept + 1:06d}"
        image_rel_path = f"kb_images/{doc_id}.jpg"
        image_abs_path = OUT_DIR / image_rel_path

        ok_image = safe_save_image(example.get("image"), image_abs_path)
        if not ok_image:
            continue

        image_url = clean_text(example.get("image_url", ""))
        metadata_url = clean_text(example.get("metadata_url", ""))

        kb[doc_id] = {
            "title": title,
            "url": page_url if page_url else metadata_url,
            "image_url": image_url,
            "metadata_url": metadata_url,
            "image_path": image_rel_path,
            "section_texts": section_texts
        }

        knn.append([doc_id])
        kept += 1

        if kept % 50 == 0:
            print(f"Creati {kept}/{TARGET_DOCS} documenti validi. Scansionati: {scanned}")

    with open(OUT_KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    with open(OUT_KNN_PATH, "w", encoding="utf-8") as f:
        json.dump(knn, f, ensure_ascii=False, indent=2)

    print("\nFATTO.")
    print(f"Documenti creati: {kept}")
    print(f"Esempi scansionati: {scanned}")
    print(f"Knowledge base: {OUT_KB_PATH}")
    print(f"Mappa KNN: {OUT_KNN_PATH}")
    print(f"Immagini KB: {KB_IMAGES_DIR}")


if __name__ == "__main__":
    main()