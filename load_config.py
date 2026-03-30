import json
from pathlib import Path

def load_config():
    base_dir = Path(__file__).resolve().parent

    with open(base_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    return {
        "model_path": base_dir / config["model_path"],
        "retriever_path": base_dir / config["retriever_path"],
        "index_path": base_dir / config["index_path"],
        "index_json_path": base_dir / config["index_json_path"],
        "kb_wikipedia_path": base_dir / config["kb_wikipedia_path"],
        "input_path": base_dir / config["input_path"],
        "output_path": base_dir / config["output_path"]
    }
    
    
    
""" 
DA INSERIRE NEL MAIN

from load_config import load_config

config = load_config()

model_path = config["model_path"]
retriever_path = config["retriever_path"]
input_path = config["input_path"]
output_path = config["output_path"]  """