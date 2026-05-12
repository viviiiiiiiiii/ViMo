"""Scarica i file di EVQA impostati nel config."""
import hashlib
import urllib.request
from pathlib import Path
import evqa_eval_config as cfg

def check_sha256(path: Path, expected: str) -> bool:
    """Calcola l'hash del file per verificare che non sia corrotto."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected.lower()

def download_file(url: str, dest: Path, expected_hash: str = None):
    """Scarica un singolo file con controlli."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists() and dest.stat().st_size > 0:
        print(f"⏭️  Già presente: {dest.name}")
        return

    print(f"⬇️  Scaricando {dest.name} da {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"✅ Download completato: {dest.name}")

    if expected_hash and not check_sha256(dest, expected_hash):
        raise ValueError(f"❌ File corrotto (SHA256 errato): {dest.name}. Cancellalo e riprova.")

def main():
    downloads =[]
    if cfg.DOWNLOAD_TRAIN_CSV: downloads.append(cfg.DATA_DIR / "train.csv")
    if cfg.DOWNLOAD_VAL_CSV: downloads.append(cfg.DATA_DIR / "val.csv")
    if cfg.DOWNLOAD_TEST_CSV: downloads.append(cfg.EVQA_TEST_FILE)
    if cfg.DOWNLOAD_KB_WIKI_ZIP: downloads.append(cfg.DATA_DIR / "encyclopedic_kb_wiki.zip")

    if not downloads:
        print("Nessun download abilitato nel config.")
        return

    for path in downloads:
        expected_hash = cfg.KB_SHA256 if path.name.endswith(".zip") else None
        download_file(cfg.URLS[path], path, expected_hash)
        
    print("\n🎉 Tutti i download necessari sono completati!")

if __name__ == "__main__":
    main()