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

# In Qwen_retrieval.py

def load_clip_and_index(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 📍 FIX 1: Forza float32 per CLIP. 
    # Risolve l'errore CUBLAS_STATUS_INTERNAL_ERROR sui nodi Boost.
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float32, 
        trust_remote_code=True
    ).to(device).eval()
    
    print("🔄 Caricamento processori CLIP...")
    from transformers import CLIPImageProcessor, AutoTokenizer
    # Usiamo il processore standard che garantisce 257 token (patch14)
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
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
    # 📍 FIX 2: Forza il calcolo in float32 per il retriever
    dtype_retriever = torch.float32 
    device = model.device

    with torch.no_grad():
        if image is not None:
            # Resize forzato per evitare mismatch di dimensioni tensor
            image_resized = image.resize((224, 224)) 
            inputs = processor.image_processor(images=image_resized, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=dtype_retriever, device=device)
            features = model.encode_image(pixel_values=pixel_values)
        elif text is not None:
            inputs = processor.tokenizer(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
            input_ids = inputs["input_ids"].to(device=device)
            # Usa la funzione nativa di CLIP per i testi
            features = model.get_text_features(input_ids=input_ids)
        
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)

def generate_answer(model, processor, messages, stop=None):
    # 📍 FIX 3: Pulizia messaggi per Qwen (rimuove tag immagine che causano assert error)
    clean_messages = []
    for m in messages:
        if isinstance(m["content"], list):
            text = " ".join([c["text"] for c in m["content"] if c["type"] == "text"])
            clean_messages.append({"role": m["role"], "content": text})
        else:
            clean_messages.append(m)

    text = processor.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    risposta = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Taglio di sicurezza per LangChain
    stop_words = stop if stop else ["Observation:"]
    for s in stop_words:
        if s in risposta: risposta = risposta.split(s)[0].strip()
            
    return risposta



def retrieve_topk_pages(features, index, index_map, wiki, k):
    _, I = index.search(features, k)
    urls = [index_map[i][0] for i in I[0]]
    texts = ["\n".join(wiki[url[0]]["section_texts"][:2]) for url in urls]
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