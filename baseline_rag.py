import torch
from PIL import Image
from Qwen_retrieval import load_clip_and_index, extract_features, retrieve_topk_pages, generate_answer
from tools_real import start_motors, qwen_model, qwen_processor, clip_model, clip_processor, knn_index_immagini, wiki_map, wiki_data
from load_config import load_config

# Inizializzazione (Ricicla i tuoi motori)
config = load_config()
class Args: pass
args = Args()
for k,v in config.items(): setattr(args, k, str(v))
args.top_k = 3

start_motors(args)

def run_multimodal_rag(image_path, question):
    # 1. Retrieval Visivo (One-Shot)
    img = Image.open(image_path).convert("RGB")
    feat = extract_features(image=img, model=clip_model, processor=clip_processor)
    context = retrieve_topk_pages(feat, knn_index_immagini, wiki_map, wiki_data, k=3)
    
    # 2. Prompt "Minestrone"
    mega_prompt = f"Usa il contesto per rispondere: {context}\n\nDomanda: {question}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": mega_prompt}
            ]
        }
    ]
    
    return generate_answer(qwen_model, qwen_processor, messages)

if __name__ == "__main__":
    res = run_multimodal_rag("foto_buia.jpg", "Quali sono le sue invenzioni?")
    print(f"Risposta RAG: {res}")