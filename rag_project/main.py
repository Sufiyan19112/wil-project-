
import json

from retriever import BM25Retriever
from generate import generate_answer
from eval import ndcg_at_k, summarize_results

K = 3  # top-k passages to retrieve, matches one of Walert's tested cutoffs


def main():
    retriever = BM25Retriever("data/passages.json")

    with open("data/test_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        retrieved = retriever.retrieve(q["question"], k=K)
        retrieved_ids = [pid for pid, _, _ in retrieved]

        answer = generate_answer(q["question"], retrieved)

        ndcg = ndcg_at_k(retrieved_ids, q["relevant_passages"], K)

        results.append({
            "question": q["question"],
            "type": q["type"],
            "retrieved_ids": retrieved_ids,
            "answer": answer,
            "ndcg": ndcg,
        })

        print(f"[{q['type']:>10}] {q['question']}")
        print(f"  retrieved: {retrieved_ids}")
        print(f"  answer:    {answer}")
        print(f"  ndcg@{K}:    {ndcg}")
        print()

    summary = summarize_results(results)
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(json.dumps(summary, indent=2))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
