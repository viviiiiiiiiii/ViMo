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

# Sostituisci QUESTE DUE FUNZIONI in Qwen_retrieval.py

# In Qwen_retrieval.py

def load_clip_and_index(args):
    device_clip = "cpu"
    print("🔄 Caricamento EVA-CLIP (Modello e Processore Ufficiali)...")
    
    # Questo è il trucco magico dal tuo build_knn_index_real.py
    import os
    os.environ["TRANSFORMERS_IGNORE_LOAD_VULNERABILITY"] = "1"
    
    from transformers import AutoModel, AutoProcessor
    
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float32, 
        trust_remote_code=True
    ).to(device_clip).eval()
    
    # 📍 Usiamo AutoProcessor invece di fare accrocchi manuali
    clip_processor = AutoProcessor.from_pretrained(args.retriever_path, trust_remote_code=True)
    
    index = faiss.read_index(str(args.index_path)) 
    with open(args.index_json_path, "r", encoding="utf-8") as f:
        index_map = json.load(f)
    with open(args.kb_wikipedia_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)  
        
    return clip_model, clip_processor, index, index_map, wiki

def extract_features(image=None, text=None, model=None, processor=None, out_dim=512):
    device = torch.device("cpu")
    with torch.no_grad():
        if image is not None:
            # 📍 IL TRUCCO: Aggiungiamo text=[""] per soddisfare l'AutoProcessor
            inputs = processor(images=image, text=[""], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            features = model.encode_image(pixel_values)
            
        elif text is not None:
            # Per il testo funziona già perfettamente
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            features = model.encode_text(input_ids)
        
        # 📍 Il ritaglio a 512 per FAISS
        if features.shape[-1] > out_dim:
            features = features[:, :out_dim]
            
        features = features / torch.clamp(features.norm(p=2, dim=-1, keepdim=True), min=1e-7)
        
    return features.float().numpy()

def generate_answer(model, processor, messages, stop=None):
    clean_messages = []
    for m in messages:
        if isinstance(m["content"], list):
            text = " ".join([c["text"] for c in m["content"] if c["type"] == "text"])
            clean_messages.append({"role": m["role"], "content": text})
        else:
            clean_messages.append(m)

    text = processor.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=True)
    
    # 📍 AGGIUNTA: Riduciamo la complessità del padding per non stressare la GPU
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, # Più corto per evitare timeout
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()



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