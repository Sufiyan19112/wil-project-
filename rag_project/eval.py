"""
Evaluation framework, following Walert's approach:
  - NDCG@k for retrieval effectiveness on Known/Inferred questions
  - % unanswered for Out-of-KB questions (did the system correctly refuse?)
"""
import math

REFUSAL_PHRASE = "i don't have information to answer that"


def dcg_at_k(relevances: list, k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def ndcg_at_k(retrieved_ids: list, relevant_passages: dict, k: int) -> float:
    """
    retrieved_ids: ordered list of passage ids returned by the retriever
    relevant_passages: {passage_id: relevance_label} e.g. {"p1": 2, "p3": 1}
    """
    if not relevant_passages:
        return None  # not applicable (e.g. out-of-KB questions)

    gains = [relevant_passages.get(pid, 0) for pid in retrieved_ids]
    dcg = dcg_at_k(gains, k)

    ideal_gains = sorted(relevant_passages.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)

    return dcg / idcg if idcg > 0 else 0.0


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE in answer.lower()


def summarize_results(results: list) -> dict:
    """
    results: list of dicts, each with keys:
        type ("known"/"inferred"/"out_of_kb"), ndcg (float or None), answer (str)
    Returns a summary dict grouped by question type.
    """
    summary = {}
    for qtype in ["known", "inferred", "out_of_kb"]:
        subset = [r for r in results if r["type"] == qtype]
        if not subset:
            continue

        entry = {"n_questions": len(subset)}

        ndcgs = [r["ndcg"] for r in subset if r["ndcg"] is not None]
        if ndcgs:
            entry["avg_ndcg"] = sum(ndcgs) / len(ndcgs)

        if qtype == "out_of_kb":
            refused = sum(1 for r in subset if is_refusal(r["answer"]))
            entry["pct_correctly_unanswered"] = round(100 * refused / len(subset), 1)

        summary[qtype] = entry

    return summary
