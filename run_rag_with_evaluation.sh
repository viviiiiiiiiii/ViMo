#!/bin/bash
#SBATCH --job-name=vimo_rag_predictions_eval
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=cvcs2026
#SBATCH --output=/homes/%u/cvcs2026/rag_predictions_eval_%j.out

# Carica l'ambiente
source /work/cvcs2026/ViMo/.venvMo/bin/activate

# Bypass per i pesi locali e vulnerabilità
export TRANSFORMERS_IGNORE_LOAD_VULNERABILITY=1

# Entra nella cartella
cd /work/cvcs2026/ViMo/

echo "=========================================="
echo "Starting RAG predictions..."
echo "=========================================="

# Esegui le previsioni RAG
python run_prediction_baseline_rag.py \
  --input /work/cvcs2026/encyclopedic/encyclopedic_test_subset.json \
  --image-root /work/cvcs2026/encyclopedic \
  --pred-out predictions_rag_test_7B.jsonl \
  --records-out records_rag_test_7B.jsonl

# Controlla se le previsioni sono riuscite
if [ $? -ne 0 ]; then
  echo "ERROR: RAG predictions failed!"
  exit 1
fi

echo "=========================================="
echo "RAG predictions completed successfully!"
echo "Starting evaluation..."
echo "=========================================="

# Esegui la valutazione
python evaluate_evqa_predictions.py \
  --gold /work/cvcs2026/encyclopedic/encyclopedic_test_subset.json \
  --pred predictions_rag_test_7B.jsonl \
  --out evqa_scores_baseline_rag_7B.json \
  --eval-utils-dir external/encyclopedic_vqa

# Controlla se la valutazione è riuscita
if [ $? -ne 0 ]; then
  echo "ERROR: Evaluation failed!"
  exit 1
fi

echo "=========================================="
echo "Evaluation completed successfully!"
echo "=========================================="
echo "Results saved in: evqa_scores_baseline_rag_7B.json"
