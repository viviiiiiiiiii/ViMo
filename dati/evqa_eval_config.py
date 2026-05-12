"""
Pannello di controllo per la valutazione su Encyclopedic-VQA (EVQA).
Modifica questo file per cambiare impostazioni, poi lancia gli script.
"""
from pathlib import Path

# --- CARTELLE E FILE BASE ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_ROOT = DATA_DIR / "images"

# --- DOWNLOAD ---
EVQA_TEST_FILE = DATA_DIR / "test.csv"
DOWNLOAD_TRAIN_CSV = False
DOWNLOAD_VAL_CSV = False
DOWNLOAD_TEST_CSV = True
DOWNLOAD_KB_WIKI_ZIP = False # Metti True solo se ti serve scaricare 4.9GB di database

URLS = {
    DATA_DIR / "train.csv": "https://storage.googleapis.com/encyclopedic-vqa/train.csv",
    DATA_DIR / "val.csv": "https://storage.googleapis.com/encyclopedic-vqa/val.csv",
    EVQA_TEST_FILE: "https://storage.googleapis.com/encyclopedic-vqa/test.csv",
    DATA_DIR / "encyclopedic_kb_wiki.zip": "https://storage.googleapis.com/encyclopedic-vqa/encyclopedic_kb_wiki.zip"
}
KB_SHA256 = "36af1b6718a975c355a776114be216f4800c61320897b2186d33d17a08e44c77"

# --- CREAZIONE SUBSET ---
SUBSET_OUTPUT_FILE = DATA_DIR / "evqa_test_singlehop_1000.jsonl"
N_QUESTIONS = 1000
RANDOM_SEED = 42
EXCLUDE_QUESTION_TYPES = {"2_hop", "two_hop", "two-hop"} # Vogliamo solo domande dirette
EXPAND_IMAGES_AS_SEPARATE_EXAMPLES = False

# --- VALUTAZIONE RAG (Normale o Agente) ---
RAG_NORMAL_PREDICTIONS_FILE = RESULTS_DIR / "rag_normal_predictions.jsonl"
AGENTIC_PREDICTIONS_FILE = RESULTS_DIR / "agentic_rag_predictions.jsonl"
METRICS_OUTPUT_PATH = RESULTS_DIR / "metrics.json" # Dove salvare i risultati finali

TOP_K = 3            # Quanti documenti recuperare per ogni domanda
EVAL_LIMIT = None    # Se vuoi testare solo su es. 5 domande, scrivi 5 qui
RESUME_IF_OUTPUT_EXISTS = True # Riprende da dove si è interrotto se salta la corrente
SAVE_RETRIEVED_CONTEXT = True

# --- SETTINGS AGENTE ---
AGENT_MODULE = "agent_real"
AGENT_FUNCTION = "agentic_rag_answer"
AGENT_BUILD_ARGS_FUNCTION = "build_args"
AGENT_LOAD_ENGINES_FUNCTION = "load_agentic_engines"

# --- METRICHE ---
USE_OFFICIAL_BEM_IF_AVAILABLE = False
PREDICTIONS_PATH = RAG_NORMAL_PREDICTIONS_FILE # Cambia questo per valutare l'agente