#!/bin/bash
#SBATCH --job-name=vimo_indexing
#SBATCH --partition=all_serial      #CAMBIATO: Chiediamo un nodo con GPU
#SBATCH --gres=gpu:1             #AGGIUNTO: Vogliamo 1 scheda video
#SBATCH --cpus-per-task=4        #AUMENTATO: Più core per caricare i dati
#SBATCH --mem=32G                #AUMENTATO: Almeno 32GB per EVA-CLIP-8B
#SBATCH --time=01:00:00          # 1 ora basta e avanza per 3 documenti
#SBATCH --account=cvcs2026
#SBATCH --output=/homes/%u/cvcs2026/index_%j.out

# Carica l'ambiente corretto (assicurati che il percorso sia giusto!)
source /work/cvcs2026/ViMo/.venvMo/bin/activate

# Entra nella cartella corretta
cd /work/cvcs2026/ViMo/

# Lancia lo script di indicizzazione
export CUDA_VISIBLE_DEVICES=""

set -e

echo "🚀 [$(date)] Inizio Pipeline EVQA"

echo "⬇️ [$(date)] STEP 1: Controllo download dataset..."
python download_evqa_dataset.py

echo "✂️ [$(date)] STEP 2: Creazione subset (1000 domande)..."
python make_evqa_subset.py

# NOTA: Scegli tu quale lanciare. In questo caso lanciamo l'AGENTE.
echo "🤖 [$(date)] STEP 3: Esecuzione Agente RAG sulle domande (Richiederà tempo)..."
python eval_agentic_rag_on_subset.py

echo "📊 [$(date)] STEP 4: Calcolo delle Metriche..."
python evaluate_answers.py

echo "🎉 [$(date)] PIPELINE COMPLETATA CON SUCCESSO!"