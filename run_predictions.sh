#!/bin/bash
#SBATCH --job-name=vimo_indexing
#SBATCH --partition=all_usr_prod      #CAMBIATO: Chiediamo un nodo con GPU
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

python run_prediction_vlm.py \
  --input /work/cvcs2026/encyclopedic/single_hop.json \
  --image-root /work/cvcs2026/encyclopedic \
  --pred-out predictions_vlm_test.jsonl \
  --records-out records_vlm_test.jsonl \
  --limit 3

  # Check: head predictions_vlm_test.jsonl  head records_vlm_test.jsonl

  python run_prediction_baseline_rag.py \
  --input /work/cvcs2026/encyclopedic/single_hop.json \
  --image-root /work/cvcs2026/encyclopedic \
  --pred-out predictions_rag_test.jsonl \
  --records-out records_rag_test.jsonl \
  --limit 3

  # Check: head predictions_rag_test.jsonl  head records_rag_test.jsonl

  python run_prediction_agentic.py \
  --input /work/cvcs2026/encyclopedic/single_hop.json \
  --image-root /work/cvcs2026/encyclopedic \
  --pred-out predictions_agentic_test.jsonl \
  --records-out records_agentic_test.jsonl \
  --limit 3

  # Check: head predictions_agentic_test.jsonl  head records_agentic_test.jsonl