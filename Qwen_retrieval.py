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
    Qwen2_5_VLForConditionalGeneration,
)
import traceback

def load_clip_and_index(args):
    # Caricamento del modello CLIP (già corretto)
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda:0").eval()
    clip_processor = AutoProcessor.from_pretrained(args.retriever_path, trust_remote_code=True)
    # Faiss richiede una stringa, quindi convertiamo il Path object in str
    index = faiss.read_index(str(args.index_path)) 
    
    # Carichiamo il mapping JSON
    with open(args.index_json_path, "r", encoding="utf-8") as f:
        index_map = json.load(f)
        
    # Carichiamo la Knowledge Base di Wikipedia
    with open(args.kb_wikipedia_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)  
        
    return clip_model, clip_processor, index, index_map, wiki



def extract_features(image=None, text=None, model=None, processor=None, out_dim=512):
    with torch.no_grad():
        if image is not None:
            # 1. RAMO VISIVO
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=model.dtype, device=model.device)
            features = model.encode_image(pixel_values=pixel_values)
            
        elif text is not None:
            # 1. Tokenizzazione con parametri di sicurezza
            inputs = processor(
                text=text, 
                return_tensors="pt", 
                padding='max_length', 
                truncation=True, 
                max_length=77  # Limite standard CLIP
            )
            # Spostiamo i dati sulla stessa periferica del modello (GPU o CPU)
            input_ids = inputs["input_ids"].to(device=model.device)
            
            # 2. TAGLIO MANUALE (La "Cintura di Sicurezza")
            # Se per qualche motivo il processor ha ignorato il max_length, 
            # forziamo noi il taglio dei tensor qui.
            if input_ids.shape[1] > 77:
                input_ids = input_ids[:, :77]
            
            # 📍 3. Esecuzione (Ora non può più crashare)
            features = model.encode_text(input_ids)
            
        else:
            raise ValueError("Devi fornire un'immagine o un testo!")

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
            max_new_tokens=64,
            do_sample=True,
            temperature=0.2,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()
