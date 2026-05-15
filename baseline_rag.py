import os
import json
from PIL import Image
import tools_real
from load_config import load_config
from Qwen_retrieval import extract_features, read_wiki_section_with_images, generate_answer

def run_standard_rag(image_path, question):
    print(f"\n🔍 [STANDARD RAG] Avvio ricerca per: {image_path}")
    
    # 1. RETRIEVE: Estrazione feature e ricerca su FAISS
    image_pil = Image.open(image_path).convert("RGB")
    features = extract_features(
        image=image_pil, 
        model=tools_real.clip_model, 
        processor=tools_real.clip_processor, 
        out_dim=512
    )
    
    # Cerchiamo solo il primissimo documento (K=1)
    k = 1
    _, I = tools_real.knn_index_immagini.search(features, k)
    url_doc_trovato = tools_real.wiki_map[I[0][0]][0]
    
    print(f"📄 Documento trovato: {url_doc_trovato}")
    
# 2. AUGMENT: Estraiamo TUTTO il testo del documento
    page = tools_real.wiki_data[url_doc_trovato]
    
    # Uniamo i titoli e i testi di tutte le sezioni
    sezioni_unite = []
    for titolo, testo in zip(page["section_titles"], page["section_texts"]):
        sezioni_unite.append(f"--- {titolo} ---\n{testo}")
    
    tutto_il_testo = "\n\n".join(sezioni_unite)
    
    # 🛡️ SCUDO ANTI-ESPLOSIONE GPU: Mettiamo un limite di sicurezza.
    # 15.000 caratteri sono circa 4000 token, più che sufficienti per rispondere 
    # senza saturare la VRAM e far crashare CUDA.
    limite_caratteri = 15000
    if len(tutto_il_testo) > limite_caratteri:
        print("⚠️ Documento lunghissimo, taglio il testo in eccesso per salvare la GPU.")
        tutto_il_testo = tutto_il_testo[:limite_caratteri] + "\n... [TESTO TRONCATO PER LIMITI DI MEMORIA]"
    
    contesto_testuale = tutto_il_testo
    
    # 3. GENERATE: Creiamo il prompt blindato per Qwen
    prompt_rag = f"""Sei un assistente AI preciso. Rispondi alla domanda finale dell'utente utilizzando ESCLUSIVAMENTE il seguente contesto tratto da Wikipedia e l'immagine fornita. Se la risposta non è nel contesto o nell'immagine, di' che non lo sai.

=== CONTESTO WIKIPEDIA ===
{contesto_testuale}
==========================

Domanda: {question}"""

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_rag}
        ]}
    ]

    print("🧠 Qwen sta generando la risposta in base al contesto...")
    
    risposta = generate_answer(
        tools_real.qwen_model, 
        tools_real.qwen_processor, 
        messages,
        max_new_tokens=256
    )
    
    return risposta

if __name__ == "__main__":
    print("🚀 Accensione motori per la Baseline RAG...")
    
    config_dict = load_config()
    class CostruttoreArgs: pass
    args = CostruttoreArgs()
    for key, value in config_dict.items():
        setattr(args, key, str(value))
    
    tools_real.start_motors(args)
    
    # Facciamo il test sulla Gioconda
    immagine_test = "foto_buia.jpg"
    domanda_test = "Chi ha dipinto quest'opera e quali sono alcune sue invenzioni famose?"
    
    risposta_finale = run_standard_rag(immagine_test, domanda_test)
    
    print("\n" + "="*50)
    print("🎯 RISPOSTA STANDARD RAG:")
    print(risposta_finale)
    print("="*50)