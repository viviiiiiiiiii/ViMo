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
    
    # 📍 TUTTO in bfloat16 per evitare conflitti CUBLAS
    tipo_dato = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=tipo_dato, 
        trust_remote_code=True
    ).to(device).eval()
    
    # Caricamento processore con fallback "corazzato"
    try:
        from transformers import CLIPImageProcessor, AutoTokenizer
        # Usiamo il processore standard che non crasha mai
        clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        clip_processor.tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)
    except:
        clip_processor = AutoProcessor.from_pretrained(args.retriever_path, trust_remote_code=True)

    # Iniezione attributi se mancano
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
    dtype = model.dtype # Usa lo stesso del modello (bfloat16)

    with torch.no_grad():
        if image is not None:
            inputs = processor.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=dtype, device=device)
            features = model.encode_image(pixel_values=pixel_values)
        elif text is not None:
            inputs = processor.tokenizer(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
            input_ids = inputs["input_ids"].to(device=device)
            # 📍 Rimosso position_ids manuale: CLIP lo genera internamente meglio di noi
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



def generate_answer(model, processor, messages):
    # Forza la pulizia dei messaggi
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Gestione sicura delle immagini
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 📍 Spostiamo tutto su GPU con il tipo di dato del modello
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        # Aumentiamo la stabilità della generazione
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False, # Meno creatività = meno errori CUDA su indici casuali
            use_cache=True
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
