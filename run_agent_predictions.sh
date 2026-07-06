#!/bin/bash
#SBATCH --job-name=vimo_test_10
#SBATCH --partition=boost_usr_prod 
#SBATCH --gres=gpu:2             
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                
#SBATCH --time=01:00:00          # Abbassato a 1 ora per il test rapido
#SBATCH --account=cvcs2026
#SBATCH --output=/homes/%u/cvcs2026/test_10_%j.out

# Carica l'ambiente
source /work/cvcs2026/ViMo/.venvMo/bin/activate
 
# Bypass per i pesi locali e vulnerabilità
export TRANSFORMERS_IGNORE_LOAD_VULNERABILITY=1

# Entra nella cartella di lavoro
cd /work/cvcs2026/ViMo/

echo "========================================="
echo "🚀 Avvio test su 10 esempi per Agentic RAG..."
echo "========================================="
python run_prediction_agentic.py \
  --input /work/cvcs2026/encyclopedic/encyclopedic_test_subset.json \
  --image-root /work/cvcs2026/encyclopedic \
  --limit 10 \
  --pred-out test10_pred_agentic.jsonl \
  --records-out test10_rec_agentic.jsonl 

echo "✅ Tutti i test completati con successo!"
