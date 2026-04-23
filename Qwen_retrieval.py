import json
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import faiss
import numpy as np
import torch
from tqdm import tqdm
from load_config import load_config
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModel,
    CLIPImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
import traceback

# In Qwen_retrieval.py

# In Qwen_retrieval.py

from transformers import AutoTokenizer, AutoImageProcessor, CLIPImageProcessor # Aggiungi questi import

# In Qwen_retrieval.py

def load_clip_and_index(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"📡 Sistema: Sto utilizzando il dispositivo -> {device}")

    # Carichiamo il modello in float32 per massima stabilità su GPU Boost
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float32, 
        trust_remote_code=True
    ).to(device).eval()
    
    print("🔄 Caricamento processori CLIP...")
    
    # 📍 SOLUZIONE DEFINITIVA: 
    # Non usiamo AutoProcessor per le immagini perché la cartella locale è corrotta.
    # Usiamo il processore standard di OpenAI che è identico per architettura Vit-L/14.
    try:
        img_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    except Exception:
        # Fallback estremo se il server non ha internet
        img_proc = CLIPImageProcessor(
            do_resize=True, size={"shortest_edge": 224}, 
            do_center_crop=True, crop_size={"height": 224, "width": 224}
        )
    
    # Il tokenizer invece lo carichiamo normalmente dai file locali
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)

    # Creiamo il wrapper per non rompere il resto del codice
    class CLIPWrapper:
        def __init__(self, ip, tk):
            self.image_processor = ip
            self.tokenizer = tk
            
    clip_processor = CLIPWrapper(img_proc, tokenizer)
        
    index = faiss.read_index(str(args.index_path)) 
    with open(args.index_json_path, "r", encoding="utf-8") as f:
        index_map = json.load(f)
    with open(args.kb_wikipedia_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)  
        
    return clip_model, clip_processor, index, index_map, wiki

def extract_features(image=None, text=None, model=None, processor=None, out_dim=512):
    # Assicuriamoci che i dati siano nello stesso formato del modello (float32)
    dtype_calc = torch.float32 
    
    with torch.no_grad():
        if image is not None:
            inputs = processor.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=dtype_calc, device=model.device)
            features = model.encode_image(pixel_values=pixel_values)
            
        elif text is not None:
            inputs = processor.tokenizer(
                text=text, 
                return_tensors="pt", 
                padding='max_length', 
                truncation=True, 
                max_length=40
            )
            input_ids = inputs["input_ids"].to(device=model.device)
            position_ids = torch.arange(40, dtype=torch.long, device=model.device).unsqueeze(0)
            
            # Esecuzione sicura
            try:
                features = model.encode_text(input_ids, position_ids=position_ids)
            except:
                outputs = model.text_model(input_ids=input_ids, position_ids=position_ids)
                features = outputs[1]
        
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def retrieve_topk_pages(features, index, index_map, wiki, k):
    _, I = index.search(features, k)
    urls = [index_map[i][0] for i in I[0]]
    texts = ["\n".join(wiki[url]["section_texts"]) for url in urls]
    return "\n\n".join(texts)

def build_chat_prompt(context, question, image):
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_content}
    ]
    return messages



def generate_answer(model, processor, messages):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()
