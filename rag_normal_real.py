import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from load_config import load_config
from Qwen_retrieval import (
    load_clip_and_index,
    extract_features,
    retrieve_topk_pages,
    generate_answer,
)


@dataclass
class RagEngines:
    """Contiene solo i motori necessari a un RAG standard.

    Nota: qui NON ci sono tool LangChain, AgentExecutor, ReAct, Action/Observation.
    Il flusso è fisso: query -> embedding -> retrieval top-k -> prompt -> generazione.
    """

    clip_model: Any
    clip_processor: Any
    index: Any
    index_map: Dict[str, Any]
    wiki: Dict[str, Any]
    qwen_model: Any
    qwen_processor: Any


def build_args(top_k: int = 3):
    config_dict = load_config()

    class Args:
        pass

    args = Args()
    for key, value in config_dict.items():
        setattr(args, key, str(value))

    args.top_k = top_k
    return args


def load_rag_engines(args) -> RagEngines:
    """Carica direttamente retriever, indice FAISS, KB e modello generativo.

    Questa funzione sostituisce tools_real.start_motors(args), così rag_normal_real.py
    non dipende dai tool dell'agente.
    """

    print("Accensione retriever CLIP + indice FAISS...")
    clip_model, clip_processor, index, index_map, wiki = load_clip_and_index(args)

    print("Accensione modello generativo Qwen2.5-VL...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    qwen_processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=True,
    ).to(device).eval()

    print("✅ RAG engines pronti")

    return RagEngines(
        clip_model=clip_model,
        clip_processor=clip_processor,
        index=index,
        index_map=index_map,
        wiki=wiki,
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
    )


def build_retrieval_query(question: str, image_path: Optional[str], engines: RagEngines):
    """Produce l'embedding della query.

    - Se c'è un'immagine, il RAG usa l'immagine come query di retrieval.
    - Se non c'è immagine, usa il testo della domanda.

    Questo è ancora RAG normale: il tipo di query è scelto dal codice, non da un agente.
    """

    if image_path is not None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Immagine non trovata: {image_path}")

        print(f"🖼️ Retrieval usando immagine come query: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return extract_features(
            image=image,
            text=None,
            model=engines.clip_model,
            processor=engines.clip_processor,
        )

    print("📝 Retrieval usando testo come query")
    return extract_features(
        image=None,
        text=question,
        model=engines.clip_model,
        processor=engines.clip_processor,
    )


def normal_rag_answer(
    question: str,
    image_path: Optional[str] = None,
    top_k: int = 3,
    engines: Optional[RagEngines] = None,
) -> Dict[str, Any]:
    """Esegue un RAG standard, senza agenti e senza tool.

    Pipeline:
    1. carica i motori, se non sono già stati passati;
    2. crea embedding della query;
    3. recupera top-k pagine dal database;
    4. passa SOLO quel contesto al modello generativo;
    5. restituisce la risposta.
    """

    args = build_args(top_k=top_k)

    if engines is None:
        engines = load_rag_engines(args)

    query_features = build_retrieval_query(question, image_path, engines)

    print(f"🔎 Retrieval top-{top_k}...")
    context = retrieve_topk_pages(
        features=query_features,
        index=engines.index,
        index_map=engines.index_map,
        wiki=engines.wiki,
        k=top_k,
    )

    prompt = f"""
You are a standard retrieval-augmented generation system.

You must answer the user question using ONLY the retrieved context below.
You do not have tools.
You cannot perform extra searches.
You cannot use hidden knowledge.
If the retrieved context is insufficient, say exactly:
"The database does not contain enough information to answer."

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()

    print("🧠 Generazione risposta grounded sul contesto...")
    answer = generate_answer(
        engines.qwen_model,
        engines.qwen_processor,
        [{"role": "user", "content": prompt}],
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.2,
    )

    return {
        "question": question,
        "image_path": image_path,
        "top_k": top_k,
        "context": context,
        "answer": answer,
    }


if __name__ == "__main__":
    result = normal_rag_answer(
        question=(
            "Identifica il soggetto in 'foto_buia.jpg'. Una volta capito chi è, usa la ricerca testuale per dirmi quali sono le sue invenzioni citate nel database che NON siano quadri."
        ),
        image_path="foto_buia.jpg",
        top_k=3,
    )

    print("\n==============================")
    print("RISPOSTA RAG NORMALE")
    print("==============================")
    print(result["answer"])
