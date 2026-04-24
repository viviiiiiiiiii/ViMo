import json
import faiss
import numpy as np
import torch
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer

def load_clip_and_index(args):
    import torch
    import os
    
    device_clip = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    dtype_clip = torch.bfloat16
    
    print(f"🔄 Caricamento EVA-CLIP su RAM per riparazione...")

    from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer
    
    # 1. Lo carichiamo in locale sulla CPU (per non far esplodere la GPU)
    clip_model = AutoModel.from_pretrained(
        args.retriever_path,
        torch_dtype=dtype_clip, 
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 📍 2. INTERVENTO CHIRURGICO: Ripariamo TUTTI gli indici di posizione corrotti
    print("⚕️ Riparazione degli indici del modello in corso...")
    for name, module in clip_model.named_modules():
        if hasattr(module, "position_ids") and module.position_ids is not None:
            shape = module.position_ids.shape
            if len(shape) == 2:
                # Sovrascriviamo l'array corrotto con un array pulito [0, 1, 2, ..., N]
                seq_len = shape[1]
                module.position_ids = torch.arange(seq_len).unsqueeze(0).to(module.position_ids.device)
    print("✅ Riparazione completata!")

    # 3. ORA lo possiamo mandare sulla GPU 1 in totale sicurezza!
    clip_model = clip_model.to(device_clip).eval()
    
    modelli_dir = os.path.dirname(str(args.retriever_path))
    local_clip_processor_path = os.path.join(modelli_dir, "clip-vit-large-patch14")
    
    print(f"🔄 Caricamento processore visivo da locale: {local_clip_processor_path}")
    
    img_proc = CLIPImageProcessor.from_pretrained(
        local_clip_processor_path, 
        local_files_only=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.retriever_path, 
        trust_remote_code=True, 
        local_files_only=True
    )
    
    class CLIPProcessorWrapper:
        def __init__(self, ip, tk):
            self.image_processor = ip
            self.tokenizer = tk
            
        def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
            if images is not None:
                return self.image_processor(images=images, return_tensors=return_tensors, **kwargs)
            if text is not None:
                return self.tokenizer(text=text, return_tensors=return_tensors, **kwargs)
                
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
    import torch
    device = model.device
    dtype = model.dtype 

    with torch.no_grad():
        if image is not None:
            # Ramo visivo (che per ora non verrà chiamato, ma lo lasciamo)
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(dtype=dtype, device=device)
            if hasattr(model, "get_image_features"):
                features = model.get_image_features(pixel_values=pixel_values)
            else:
                features = model.encode_image(pixel_values=pixel_values)
            
        elif text is not None:
            print(f"\n🔍 [DEBUG TESTO] Elaborazione query: '{text}'")
            
            # 📍 IL FIX: Niente più padding! Lasciamo la frase della sua lunghezza naturale.
            inputs = processor.tokenizer(
                text=text,
                return_tensors="pt",
                truncation=True
                # RIMOSSO: padding='max_length' e max_length=40
            )
            
            input_ids = inputs["input_ids"].to(device)
            print(f"📊 [DEBUG TESTO] Forma input_ids NATURALE: {input_ids.shape}")
            
            # 📍 IL LUCCHETTO MATEMATICO: Forziamo ogni singolo token a stare 
            # rigorosamente dentro i limiti del vocabolario CLIP (0 - 49407).
            # Questo impedisce FISICAMENTE alla GPU di andare in "out of bounds".
            input_ids = torch.clamp(input_ids, min=0, max=49407)
            
            print(f"🚀 [DEBUG TESTO] Lancio il modello per estrarre le features testuali...")
            
            if hasattr(model, "get_text_features"):
                features = model.get_text_features(input_ids=input_ids)
            else:
                features = model.encode_text(input_ids)
                
            print(f"✅ [DEBUG TESTO] Estrazione riuscita! Forma: {features.shape}")
        
            
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