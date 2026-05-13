import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from Qwen_retrieval import generate_answer
from load_config import load_config

# Caricamento configurazione
config = load_config()

print("🚀 Caricamento Plain VLM...")
processor = AutoProcessor.from_pretrained(config['model_path'], local_files_only=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map="cuda:0"
).eval()

def run_vlm_only(image_path, question):
    # Messaggio multimodale diretto
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Generazione risposta (senza aiuti esterni)
    return generate_answer(model, processor, messages, max_new_tokens=256)

if __name__ == "__main__":
    # Test rapido
    res = run_vlm_only("foto_buia.jpg", "Chi è l'autore e quali sono le sue invenzioni?")
    print(f"Risposta VLM: {res}")