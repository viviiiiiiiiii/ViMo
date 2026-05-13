import json
import csv
import time
import os

# 📍 Importiamo i tre "Sfidanti" dai file che hai creato
try:
    from baseline_vlm import run_vlm_only
    from baseline_rag import run_multimodal_rag
    from agent_real import run_agentic_rag, tools_real, args # Assicurati di esporre args e tools per i motori
except ImportError as e:
    print(f"⚠️ Attenzione: Assicurati che i file delle baseline siano nella stessa cartella. Errore: {e}")

# ==========================================
# 1. SETUP DEL DATASET DI TEST
# ==========================================
# Per ora creiamo un dataset finto. 
# Quando avrai il dataset vero (OK-VQA), farai: dataset = json.load(open('okvqa_subset.json'))
dataset_di_test = [
    {
        "id": 1,
        "image": "foto_buia.jpg",
        "question": "Identifica l'autore di questo quadro e dimmi quali sono le sue invenzioni meccaniche citate nel database.",
        "ground_truth": "Leonardo da Vinci. Vite aerea, carro armato, paracadute, carro semovente."
    },
    {
        "id": 2,
        "image": "foto_buia.jpg", 
        "question": "In che anno è stata rubata quest'opera e da chi?",
        "ground_truth": "1911, da Vincenzo Peruggia."
    }
    # Aggiungi qui altre domande...
]

# ==========================================
# 2. FUNZIONE PRINCIPALE DI VALUTAZIONE
# ==========================================
def esegui_valutazione(dataset, output_csv="risultati_valutazione.csv"):
    print(f"🚀 Inizio valutazione su {len(dataset)} domande...")
    
    # Apriamo il file CSV per salvare i risultati man mano (modalità append/write)
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';') # Uso il punto e virgola per evitare casini con le virgole nel testo
        
        # Intestazione delle colonne
        writer.writerow([
            "ID", "Immagine", "Domanda", "Risposta Attesa (Ground Truth)", 
            "Risposta PLAIN VLM", "Tempo VLM (s)", 
            "Risposta RAG (Non-Agentic)", "Tempo RAG (s)", 
            "Risposta AGENTIC RAG", "Tempo Agente (s)",
            "Esito Agente (Manuale)"
        ])
        
        for item in dataset:
            img_path = item["image"]
            question = item["question"]
            gt = item["ground_truth"]
            
            print(f"\n" + "="*50)
            print(f"📝 TEST ID: {item['id']} | Domanda: {question}")
            
            # --- SFIDANTE 1: PLAIN VLM ---
            print("🤖 Avvio Plain VLM...")
            start_time = time.time()
            try:
                ans_vlm = run_vlm_only(img_path, question)
            except Exception as e:
                ans_vlm = f"ERRORE: {str(e)}"
            time_vlm = round(time.time() - start_time, 2)
            
            # --- SFIDANTE 2: NON-AGENTIC RAG ---
            print("🔍 Avvio Multimodal RAG (Baseline)...")
            start_time = time.time()
            try:
                ans_rag = run_multimodal_rag(img_path, question)
            except Exception as e:
                ans_rag = f"ERRORE: {str(e)}"
            time_rag = round(time.time() - start_time, 2)
            
            # --- SFIDANTE 3: AGENTIC RAG ---
            print("🕵️‍♂️ Avvio Agentic RAG...")
            start_time = time.time()
            try:
                ans_agent = run_agentic_rag(img_path, question)
            except Exception as e:
                ans_agent = f"ERRORE: {str(e)}"
            time_agent = round(time.time() - start_time, 2)
            
            # Salvataggio nel CSV
            writer.writerow([
                item['id'], img_path, question, gt,
                ans_vlm, time_vlm,
                ans_rag, time_rag,
                ans_agent, time_agent,
                "" # Colonna vuota per i tuoi appunti manuali
            ])
            
            # Forza la scrittura sul disco (se il server crasha, non perdi i dati!)
            file.flush()
            
            print(f"✅ Test {item['id']} salvato con successo.")

    print(f"\n🎉 Valutazione completata! Risultati salvati in: {output_csv}")

# ==========================================
# 3. AVVIO
# ==========================================
if __name__ == "__main__":
    # Assicurati che i motori siano accesi prima di iniziare!
    # tools_real.start_motors(args) # <-- Decommenta e adatta se serve inizializzare tutto da qui
    
    esegui_valutazione(dataset_di_test)