"""
Standalone retrieval evaluation script — pure Python BM25 implementation
(no external deps needed), for environments without pip/network access.
Computes real NDCG@k on the current data/passages.json + test_questions.json.
"""
import json
import math
import re
from collections import Counter

K = 3

def tokenize(text):
    return re.findall(r"\w+", text.lower())

class SimpleBM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.corpus = corpus_tokens
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_lens = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.N
        self.df = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        self.idf = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in self.df.items()
        }
        self.tf = [Counter(doc) for doc in corpus_tokens]

    def score(self, query_tokens, idx):
        score = 0.0
        doc_len = self.doc_lens[idx]
        for term in query_tokens:
            if term not in self.idf:
                continue
            freq = self.tf[idx].get(term, 0)
            numer = self.idf[term] * freq * (self.k1 + 1)
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += numer / denom if denom > 0 else 0
        return score

    def get_scores(self, query_tokens):
        return [self.score(query_tokens, i) for i in range(self.N)]


def dcg_at_k(relevances, k):
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

def ndcg_at_k(retrieved_ids, relevant_passages, k):
    if not relevant_passages:
        return None
    gains = [relevant_passages.get(pid, 0) for pid in retrieved_ids]
    dcg = dcg_at_k(gains, k)
    ideal = sorted(relevant_passages.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def main():
    with open("data/passages.json") as f:
        passages = json.load(f)
    with open("data/test_questions.json") as f:
        questions = json.load(f)

    ids = [p["id"] for p in passages]
    texts = [p["text"] for p in passages]
    corpus_tokens = [tokenize(t) for t in texts]
    bm25 = SimpleBM25(corpus_tokens)

    print(f"Knowledge base: {len(passages)} passages")
    print(f"Test set: {len(questions)} questions\n")
    print(f"{'Type':<10} {'Question':<65} {'Top retrieved':<10} {'NDCG@3'}")
    print("-" * 100)

    results = []
    for q in questions:
        query_tokens = tokenize(q["question"])
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(ids, texts, scores), key=lambda x: x[2], reverse=True)[:K]
        retrieved_ids = [r[0] for r in ranked]

        ndcg = ndcg_at_k(retrieved_ids, q["relevant_passages"], K)
        results.append({"type": q["type"], "question": q["question"],
                         "retrieved_ids": retrieved_ids, "ndcg": ndcg})

        ndcg_str = f"{ndcg:.3f}" if ndcg is not None else "N/A"
        print(f"{q['type']:<10} {q['question'][:63]:<65} {str(retrieved_ids):<10} {ndcg_str}")

    print("\n" + "=" * 50)
    print("SUMMARY BY QUESTION TYPE")
    print("=" * 50)
    for qtype in ["known", "inferred", "out_of_kb"]:
        subset = [r for r in results if r["type"] == qtype]
        if not subset:
            continue
        ndcgs = [r["ndcg"] for r in subset if r["ndcg"] is not None]
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else None
        print(f"{qtype:<12} n={len(subset)}  avg NDCG@{K}={avg_ndcg:.3f}" if avg_ndcg is not None
              else f"{qtype:<12} n={len(subset)}  avg NDCG@{K}=N/A (no relevance judgments)")

    with open("retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full results to retrieval_results.json")

if __name__ == "__main__":
    main()
