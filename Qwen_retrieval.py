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

# In Qwen_retrieval.py

def load_clip_and_index(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tipo_dato = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Caricamento Modello
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=tipo_dato, 
        trust_remote_code=True
    ).to(device).eval()
    
    # 📍 FIX RESOLUTION: Forziamo la risoluzione a 224 per evitare il mismatch 577/257
    from transformers import CLIPImageProcessor, AutoTokenizer
    try:
        # Usiamo il processore base di OpenAI (Vit-L/14) che genera esattamente 257 token
        clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    except:
        # Fallback manuale se il server è offline
        clip_processor = CLIPImageProcessor(
            do_resize=True, size={"shortest_edge": 224}, 
            do_center_crop=True, crop_size={"height": 224, "width": 224}
        )
    
    clip_processor.tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)

    if not hasattr(clip_processor, "image_processor"): clip_processor.image_processor = clip_processor
    if not hasattr(clip_processor, "tokenizer"): clip_processor.tokenizer = clip_processor
        
    index = faiss.read_index(str(args.index_path)) 
    with open(args.index_json_path, "r", encoding="utf-8") as f:
        index_map = json.load(f)
    with open(args.kb_wikipedia_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)  
        
    return clip_model, clip_processor, index, index_map, wiki

def extract_features(image=None, text=None, model=None, processor=None, out_dim=512):
    device = model.device
    dtype = model.dtype # bfloat16 su Boost

    with torch.no_grad():
        if image is not None:
            # 📍 FORZIAMO IL RESIZE: EVA-CLIP-8B vuole 224x224 per avere 257 token
            image_resized = image.resize((224, 224)) 
            inputs = processor.image_processor(images=image_resized, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=dtype, device=device)
            features = model.encode_image(pixel_values=pixel_values)
        elif text is not None:
            # CLIP text encoder standard
            inputs = processor.tokenizer(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
            input_ids = inputs["input_ids"].to(device=device)
            features = model.get_text_features(input_ids=input_ids)
        
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



# In Qwen_retrieval.py

def generate_answer(model, processor, messages, stop=None):
    # 1. Preparazione testo
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 2. Tokenizzazione
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    # 3. Configurazione Stop Tokens (fondamentale per ReAct)
    # Se LangChain ci passa dei token di stop, li usiamo, altrimenti usiamo Observation
    stop_words = stop if stop else ["Observation:"]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False, # Deterministico per evitare errori di formato
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    risposta = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Tagliamo la risposta se il modello ha ignorato lo stop e ha scritto "Observation:"
    for word in stop_words:
        if word in risposta:
            risposta = risposta.split(word)[0].strip()
            
    return risposta
