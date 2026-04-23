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
    # Logica multi-GPU come nel tuo file originale
    device_clip = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
    print(f"🔄 Caricamento EVA-CLIP su {device_clip} (Protezione File Attiva)...")
    
    import os
    os.environ["TRANSFORMERS_IGNORE_LOAD_VULNERABILITY"] = "1"
    
    from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer
    
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).to(device_clip).eval()
    
    # 📍 IL PARACADUTE: Se manca il file json, creiamo il processore a mano
    try:
        img_proc = CLIPImageProcessor.from_pretrained(args.retriever_path)
    except OSError:
        print("⚠️ File di configurazione visiva non trovato, applico parametri di sicurezza...")
        img_proc = CLIPImageProcessor(
            do_resize=True, 
            size={"shortest_edge": 224}, 
            do_center_crop=True, 
            crop_size={"height": 224, "width": 224}
        )
        
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_path, trust_remote_code=True)
    
    class CLIPProcessorWrapper:
        def __init__(self, ip, tk):
            self.image_processor = ip
            self.tokenizer = tk
            
    clip_processor = CLIPProcessorWrapper(img_proc, tokenizer)
    
    import faiss
    import json
    index = faiss.read_index(str(args.index_path)) 
    with open(args.index_json_path, "r", encoding="utf-8") as f:
        index_map = json.load(f)
    with open(args.kb_wikipedia_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)  
        
    return clip_model, clip_processor, index, index_map, wiki

def extract_features(image=None, text=None, model=None, processor=None, out_dim=512):
    device = model.device 
    
    with torch.no_grad():
        if image is not None:
            # Esattamente la tua logica originale dei tensori
            inputs = processor.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=torch.float16, device=device)
            features = model.encode_image(pixel_values=pixel_values)
            
        elif text is not None:
            try:
                # Proviamo con max_length=64, limite classico per alcuni modelli EVA
                inputs = processor.tokenizer(
                    text=text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=64 
                )
                input_ids = inputs["input_ids"].to(device)
                features = model.encode_text(input_ids)
            except Exception as e:
                # Se il modello testuale è corrotto, restituiamo un errore pulito, non un Traceback
                raise ValueError(f"Codificatore testuale non supportato per questa query.")
            input_ids = inputs["input_ids"].to(device)
            features = model.encode_text(input_ids)
        
        # Taglio a 512 per FAISS
        if features.shape[-1] > out_dim:
            features = features[:, :out_dim]
            
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        
    return features.cpu().numpy().astype(np.float32)

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