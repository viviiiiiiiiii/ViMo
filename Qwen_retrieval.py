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
    device_clip = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
    print(f"🔄 Caricamento EVA-CLIP su {device_clip}...")
    
    import os
    os.environ["TRANSFORMERS_IGNORE_LOAD_VULNERABILITY"] = "1"
    
    from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer
    import math
    
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ).to(device_clip).eval()
    
    # 📍 LA MAGIA ASSOLUTA: Leggiamo la dimensione esatta dai neuroni del modello!
    try:
        pos_weight = clip_model.vision_model.embeddings.position_embedding.weight
        num_patches = pos_weight.shape[0] - 1  # Sottraiamo 1 per il token CLS
        true_size = int(math.sqrt(num_patches) * 14)
        print(f"\n🔮 [DEBUG] LETTURA NEURONI: La matrice ha {pos_weight.shape[0]} posizioni.")
        print(f"✅ [DEBUG] Dimensione immagine richiesta dal modello: {true_size}x{true_size}!\n")
    except Exception as e:
        print(f"\n⚠️ [DEBUG] Impossibile leggere i neuroni ({e}), forzo 336x336 di sicurezza.\n")
        true_size = 336
        
    img_proc = CLIPImageProcessor(
        do_resize=True, 
        size={"shortest_edge": true_size}, 
        do_center_crop=True, 
        crop_size={"height": true_size, "width": true_size}
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
    
    print("\n" + "="*40)
    print("🕵️ DEBUG PROFONDO TENSORI EVA-CLIP")
    print("="*40)

    with torch.no_grad():
        if image is not None:
            print("▶ TIPO INPUT: Immagine")
            
            inputs = processor.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=torch.float16, device=device)
            
            print(f"📦 [DEBUG] Forma del tensore inviato (pixel_values): {pixel_values.shape}")
            print("🚀 [DEBUG] Lancio model.encode_image()...")
            
            features = model.encode_image(pixel_values=pixel_values)
            print("✅ [DEBUG] SUCCESSO! Immagine processata.")
            
        elif text is not None:
            print(f"▶ TIPO INPUT: Testo -> '{text}'")
            
            # Testo dinamico, niente più forzature che fanno crashare i bordi
            inputs = processor.tokenizer(
                text=text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True 
            )
            input_ids = inputs["input_ids"].to(device)
            
            print(f"📦 [DEBUG] Forma del tensore inviato (input_ids): {input_ids.shape}")
            print(f"🔢 [DEBUG] Token ID massimo inviato: {input_ids.max().item()}")
            print("🚀 [DEBUG] Lancio model.encode_text()...")
            
            features = model.encode_text(input_ids)
            print("✅ [DEBUG] SUCCESSO! Testo processato.")
        
        # Taglio a 512 per FAISS
        if features.shape[-1] > out_dim:
            print(f"✂️ [DEBUG] Taglio features da {features.shape[-1]} a {out_dim} dimensioni per FAISS.")
            features = features[:, :out_dim]
            
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        print(f"🏁 [DEBUG] Feature estratte pronte! (forma: {features.shape})")
        print("="*40 + "\n")
        
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