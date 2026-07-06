#!/bin/bash
#SBATCH --job-name=vimo_test_2
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cvcs2026
#SBATCH --output=/homes/%u/cvcs2026/agentic_test_2_%j.out

# 1. Carica l'ambiente
source /work/cvcs2026/ViMo/.venvMo/bin/activate
 
# 2. Setup e Pulizia Cache
export TRANSFORMERS_IGNORE_LOAD_VULNERABILITY=1
# Pulizia della cache immagini per evitare di finire lo spazio
mkdir -p /work/cvcs2026/ViMo/tmp_wiki_images
rm -rf /work/cvcs2026/ViMo/tmp_wiki_images/*

# 3. Entra nella cartella di lavoro
cd /work/cvcs2026/ViMo/

echo "======================================================"
echo "🚀 STEP 1: Avvio PREDIZIONI Agentic RAG (Test 2)"
echo "======================================================"
export QWEN_MODEL_PATH="modelli/Qwen2.5-VL-3B-Instruct"

python run_prediction_agentic_2.py \
  --input /work/cvcs2026/encyclopedic/encyclopedic_test_subset.json \
  --image-root /work/cvcs2026/encyclopedic \
  --pred-out predictions_agentic_19.jsonl \
  --records-out records_agentic_19.jsonl 

echo "======================================================"
echo "📊 STEP 2: Avvio VALUTAZIONE sui risultati generati"
echo "======================================================"

python evaluate_evqa_predictions.py \
  --gold /work/cvcs2026/encyclopedic/encyclopedic_test_subset.json \
  --pred predictions_agentic_19.jsonl \
  --out evqa_scores_agentic_19.json \
  --eval-utils-dir external/encyclopedic_vqa

echo "======================================================"
echo "✅ JOB COMPLETATO! File generato: evqa_scores_agentic_19.json"
echo "======================================================"