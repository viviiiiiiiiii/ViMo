#!/bin/bash
#SBATCH --job-name=vimo_predictions_test
#SBATCH --partition=all_usr_prod  # La partizione GPU che abbiamo trovato
#SBATCH --gres=gpu:2             #Chiediamo 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                # Qwen2.5-VL ha bisogno di molta RAM
#SBATCH --time=00:20:00
#SBATCH --account=cvcs2026
#SBATCH --output=/homes/%u/cvcs2026/evaluation_%j.out


# Carica l'ambiente
source /work/cvcs2026/ViMo/.venvMo/bin/activate
 
# Bypass per i pesi locali e vulnerabilità
export TRANSFORMERS_IGNORE_LOAD_VULNERABILITY=1

# Entra nella cartella ed esegui l'agente
cd /work/cvcs2026/ViMo/


python evaluate_evqa_predictions.py \
  --gold /work/cvcs2026/encyclopedic/encyclopedic_test_subset.json \
  --pred test10_pred_agentic.jsonl \
  --out evqa_scores_agentic_10.json   \
  --eval-utils-dir external/encyclopedic_vqa \
  --limit 10