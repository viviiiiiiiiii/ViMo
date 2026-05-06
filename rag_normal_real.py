import argparse
from PIL import Image

import tools_real
from load_config import load_config
from Qwen_retrieval import (
    extract_features,
    retrieve_topk_pages,
    generate_answer
)


def build_args():
    config_dict = load_config()

    class Args:
        pass

    args = Args()
    for key, value in config_dict.items():
        setattr(args, key, str(value))

    args.top_k = 3
    return args


def normal_rag_answer(question, image_path=None, top_k=3):
    args = build_args()
    args.top_k = top_k

    print("🚀 Inizializzazione motori condivisi...")
    tools_real.start_motors(args)

    if image_path is not None:
        print(f"🖼️ Uso immagine come query: {image_path}")
        image = Image.open(image_path).convert("RGB")

        features = extract_features(
            image=image,
            model=tools_real.clip_model,
            processor=tools_real.clip_processor
        )

    else:
        print("📝 Uso testo come query.")

        features = extract_features(
            text=question,
            model=tools_real.clip_model,
            processor=tools_real.clip_processor
        )

    print("🔎 Retrieval top-k...")

    context = retrieve_topk_pages(
        features,
        tools_real.index,
        tools_real.index_map,
        tools_real.wiki,
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
        tools_real.qwen_model,
        tools_real.qwen_processor,
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

    result = normal_rag_answer(
        question="Usando il contesto recuperato dall'immagine, identifica il soggetto e dimmi quali invenzioni non artistiche sono citate nel database.",
        image_path="foto_buia.jpg",
        top_k=3
    )

    print("\n==============================")
    print("RISPOSTA RAG NORMALE")
    print("==============================")
    print(result["answer"])