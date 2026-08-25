"""
Generation stage: send the question + top-k retrieved passages to a local
Ollama model and get back a synthesized answer.

Requires Ollama running locally (https://ollama.com/) with a model pulled, e.g.:
    ollama pull llama3.2
    ollama serve   (usually starts automatically)
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # swap for whatever model you've pulled

PROMPT_TEMPLATE = """You are answering a question using ONLY the retrieved passages below.
If the passages do not contain enough information to answer the question, respond
exactly with: "I don't have information to answer that." Do not make anything up.

Question: {question}

Retrieved passages:
{passages}

Answer:"""


def generate_answer(question: str, retrieved_passages: list, model: str = MODEL_NAME) -> str:
    """
    retrieved_passages: list of (passage_id, text, score) from the retriever
    """
    passages_block = "\n".join(f"- {text}" for _, text, _ in retrieved_passages)
    prompt = PROMPT_TEMPLATE.format(question=question, passages=passages_block)

    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


if __name__ == "__main__":
    from retriever import BM25Retriever

    r = BM25Retriever("data/passages.json")
    q = "How much sunlight do tomatoes need?"
    top_passages = r.retrieve(q, k=3)
    print(generate_answer(q, top_passages))
