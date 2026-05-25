import os
import json
from PIL import Image
import tools_real
from load_config import load_config
from Qwen_retrieval import extract_features, read_wiki_section_with_images, generate_answer

from eval_utils import (
    build_common_record, elapsed, now_seconds,
    parse_retrieved_urls, parse_sections, retrieval_metrics, token_estimate,
)

# Inizializzazione
config = load_config()
class Args: pass
args = Args()
for k, v in config.items():
    setattr(args, k, str(v))
args.top_k = 3

tools_real.start_motors(args)

def run_standard_rag(image_path, question, return_metadata=False, question_id=None, ground_truth="", question_type="unknown", expected_sources=None, top_k=3,):
    print(f"\n🔍 [STANDARD RAG] Avvio ricerca per: {image_path}")
    
    start = now_seconds()
    error = None
    context = ""
    retrieved_urls = []
    sections =[]
    answer = ""
    # Cerchiamo solo il primissimo documento (K=1)
    k = 1
    
    try:
        # 1. RETRIEVE: Estrazione feature e ricerca su FAISS
        image_pil = Image.open(image_path).convert("RGB")
        features = extract_features(
            image=image_pil, 
            model=tools_real.clip_model, 
            processor=tools_real.clip_processor, 
            out_dim=512
        )
        
        _, I = tools_real.knn_index_immagini.search(features, k)
        url_doc_trovato = tools_real.wiki_map[I[0][0]][0]
        retrieved_urls = [url_doc_trovato]
        
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
        sections = page.get("section_titles", [])
        context = contesto_testuale
        
        # 3. GENERATE: Creiamo il prompt blindato per Qwen
        prompt_rag = f"""Try to answer the question based on the retrieved Wikipedia context. If the context does not contain the answer, do your best to provide a plausible answer based on the information available.

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
        
        answer = generate_answer(
            tools_real.qwen_model, 
            tools_real.qwen_processor, 
            messages,
            max_new_tokens=256
        )
    except Exception as e:
        error = str(e)
        answer = f"ERRORE RAG: {error}"

    latency = elapsed(start)

    if not return_metadata:
        return answer

    ret_metrics = retrieval_metrics(retrieved_urls, expected_sources, k=k)
    record = build_common_record(
        question_id=question_id,
        model_name="standard_rag",
        image_path=image_path,
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        question_type=question_type,
        latency_seconds=latency,
        error=error,
        extra={
            "has_retrieval": 1,
            "retrieval_mode": "visual_k1",
            "top_k": k,
            "retrieved_urls": retrieved_urls,
            "retrieved_sections": sections,
            "retrieved_context_chars": len(context),
            "context_tokens_est": token_estimate(context),
            "num_steps": 2,
            "num_tool_calls": 0,
            "num_retrieval_calls": 1,
            "visual_input_used": 1,
            **ret_metrics,
        }
    )
    return record

if __name__ == "__main__":
    print("🚀 Accensione motori per la Baseline RAG...")
    
    # Facciamo il test sulla Gioconda
    immagine_test = "foto_buia.jpg"
    domanda_test = "Chi ha dipinto quest'opera e quali sono alcune sue invenzioni famose?"
    
    risposta_finale = run_standard_rag(immagine_test, domanda_test)
    
    print("\n" + "="*50)
    print("🎯 RISPOSTA STANDARD RAG:")
    print(risposta_finale)
    print("="*50)