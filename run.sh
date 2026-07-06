#!/bin/bash
#SBATCH --job-name=vimo_agentic_full
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cvcs2026
#SBATCH --output=/homes/%u/cvcs2026/agentic_full_%j.out

# 1. Carica l'ambiente
source /work/cvcs2026/ViMo/.venvMo/bin/activate
 
# 2. Setup e Pulizia Cache
export TRANSFORMERS_IGNORE_LOAD_VULNERABILITY=1
# Aggiungi questo per evitare che accumuli migliaia di file in 24 ore
mkdir -p /work/cvcs2026/ViMo/tmp_wiki_images
rm -rf /work/cvcs2026/ViMo/tmp_wiki_images/*

# 3. Entra nella cartella di lavoro
cd /work/cvcs2026/ViMo/

echo "======================================================"
echo "🚀 STEP 1: Avvio PREDIZIONI Agentic RAG (Full Dataset)"
echo "======================================================"

export QWEN_MODEL_PATH="modelli/Qwen3-VL-8B-Instruct"

python run_prediction_agentic.py \
  --input /work/cvcs2026/encyclopedic/encyclopedic_test_subset.json \
  --image-root /work/cvcs2026/encyclopedic \
  --pred-out predictions_agentic_full_21.jsonl \
  --records-out records_agentic_full_21.jsonl 

echo "======================================================"
echo "📊 STEP 2: Avvio VALUTAZIONE sui risultati generati"
echo "======================================================"

python evaluate_evqa_predictions.py \
  --gold /work/cvcs2026/encyclopedic/encyclopedic_test_subset.json \
  --pred predictions_agentic_full_21.jsonl \
  --out evqa_scores_agentic_full_21.json \
  --eval-utils-dir external/encyclopedic_vqa

echo "======================================================"
echo "✅ JOB COMPLETATO! File generato: evqa_scores_agentic_full_21.json"
echo "======================================================"