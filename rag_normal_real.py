import os
import argparse
from PIL import Image

from load_config import load_config
from Qwen_retrieval import (
    load_clip_and_index,
    extract_features,
    retrieve_topk_pages,
    generate_answer
)

from transformers import AutoModelForVision2Seq, AutoProcessor
import torch


def build_args():
    config_dict = load_config()

    class Args:
        pass

    args = Args()
    for key, value in config_dict.items():
        setattr(args, key, str(value))

    args.top_k = 3
    return args


def load_qwen_generator(args):
    """
    Carica il modello generativo Qwen.
    Adatta i nomi dei path se nel vostro config sono diversi.
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"🔄 Caricamento modello generativo su {device}...")

    model = AutoModelForVision2Seq.from_pretrained(
        args.generator_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    ).eval()

    processor = AutoProcessor.from_pretrained(
        args.generator_path,
        trust_remote_code=True,
        local_files_only=True
    )

    return model, processor


def normal_rag_answer(question, image_path=None, top_k=3):
    args = build_args()
    args.top_k = top_k

    print("🚀 Caricamento retriever + indice FAISS...")
    clip_model, clip_processor, index, index_map, wiki = load_clip_and_index(
        args,
        load_faiss=True
    )

    print("🚀 Caricamento modello generativo...")
    qwen_model, qwen_processor = load_qwen_generator(args)

    if image_path is not None:
        print(f"🖼️ Uso immagine come query: {image_path}")
        image = Image.open(image_path).convert("RGB")

        features = extract_features(
            image=image,
            model=clip_model,
            processor=clip_processor
        )

    else:
        print("📝 Uso testo come query.")
        features = extract_features(
            text=question,
            model=clip_model,
            processor=clip_processor
        )

    print("🔎 Retrieval top-k...")
    context = retrieve_topk_pages(
        features,
        index,
        index_map,
        wiki,
        k=top_k
    )

    prompt = f"""
You are a standard RAG assistant.

Answer the user question using ONLY the provided context.
Do not use tools.
Do not invent information.
If the answer is not present in the context, say that the database does not contain enough information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    print("🧠 Generazione risposta...")
    answer = generate_answer(
        qwen_model,
        qwen_processor,
        messages,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.2
    )

    return {
        "question": question,
        "image_path": image_path,
        "context": context,
        "answer": answer
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3)

    args = parser.parse_args()

    result = normal_rag_answer(
        question= "Identifica il soggetto in 'foto_buia.jpg'. Una volta capito chi è, usa la ricerca testuale per dirmi quali sono le sue invenzioni citate nel database che NON siano quadri.",
        image_path="foto_buia.jpg",
        top_k=3
    )

    print("\n==============================")
    print("RISPOSTA RAG NORMALE")
    print("==============================")
    print(result["answer"])