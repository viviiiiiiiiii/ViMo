import json
import faiss
import numpy as np
import torch
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer

def load_clip_and_index(args):
    import torch
    # CLIP va sulla GPU 1, Qwen resta al sicuro sulla GPU 0
    device_clip = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    dtype_clip = torch.bfloat16
    
    print(f"🔄 Caricamento EVA-CLIP su {device_clip}...")

    from transformers import AutoModel, AutoProcessor, AutoImageProcessor, AutoTokenizer
    
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=dtype_clip, 
        trust_remote_code=True
    ).to(device_clip).eval()
    
    print("🔄 Caricamento processori NATIVI del modello...")
    # 📍 IL TUO INTUITO: Non forziamo più nulla a mano.
    # Carichiamo la configurazione esatta e nativa del modello!
    try:
        clip_processor = AutoProcessor.from_pretrained(args.retriever_path, trust_remote_code=True)
    except Exception:
        # Fallback pulito se manca l'AutoProcessor unificato
        class NativeProcessorWrapper:
            def __init__(self):
                self.image_processor = AutoImageProcessor.from_pretrained(args.retriever_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(args.retriever_path, trust_remote_code=True)
        clip_processor = NativeProcessorWrapper()
        
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
    dtype = model.dtype 

    with torch.no_grad():
        if image is not None:
            # Estraiamo con il processore NATIVO senza alterare le taglie
            if hasattr(processor, "image_processor"):
                inputs = processor.image_processor(images=image, return_tensors="pt")
            else:
                inputs = processor(images=image, return_tensors="pt")
                
            pixel_values = inputs["pixel_values"].to(dtype=dtype, device=device)
            
            if hasattr(model, "get_image_features"):
                features = model.get_image_features(pixel_values=pixel_values)
            else:
                features = model.encode_image(pixel_values=pixel_values)
            
        elif text is not None:
            # 📍 IL FIX ASSOLUTO: Ora estraiamo sia gli input_ids che l'ATTENTION_MASK!
            if hasattr(processor, "tokenizer"):
                inputs = processor.tokenizer(text=text, return_tensors="pt", padding=True, truncation=True)
            else:
                inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
                
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            if hasattr(model, "get_text_features"):
                if attention_mask is not None:
                    features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    features = model.get_text_features(input_ids=input_ids)
            else:
                # Metodo standard per le varianti custom di EVA-CLIP
                features = model.encode_text(input_ids)
        
        if features.shape[-1] > out_dim:
            features = features[:, :out_dim]
            
        features = features / torch.clamp(features.norm(p=2, dim=-1, keepdim=True), min=1e-7)
        
    return features.to(torch.float32).cpu().numpy()


def generate_answer(model, processor, messages, stop=None):
    clean_messages = []
    for m in messages:
        if isinstance(m["content"], list):
            text = " ".join([c["text"] for c in m["content"] if c["type"] == "text"])
            clean_messages.append({"role": m["role"], "content": text})
        else:
            clean_messages.append(m)

    text = processor.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128,
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