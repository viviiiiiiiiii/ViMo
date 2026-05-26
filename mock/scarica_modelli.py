# scarica_modelli.py
from huggingface_hub import snapshot_download

print("📥 Inizio scaricamento di Qwen2.5-VL (Potrebbe volerci un po')...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir="/work/cvcs2026/ViMo/modelli/Qwen2.5-VL-3B-Instruct",
    local_dir_use_symlinks=False  # 🚀 FONDAMENTALE: Scarica i file veri, non i collegamenti!
)

print("📥 Inizio scaricamento di CLIP...")
snapshot_download(
    repo_id="openai/clip-vit-large-patch14",
    local_dir="/work/cvcs2026/ViMo/modelli/clip-vit-large-patch14",
    local_dir_use_symlinks=False
)

print("✅ Modelli scaricati perfettamente in locale!")