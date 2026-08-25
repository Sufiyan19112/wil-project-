"""
BM25 retriever for the RAG pipeline.
Mirrors the retrieval stage in Walert (Okapi BM25 with default params).
"""
import json
import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    def __init__(self, passages_path: str):
        with open(passages_path, "r", encoding="utf-8") as f:
            self.passages = json.load(f)  # list of {"id": ..., "text": ...}

        self.ids = [p["id"] for p in self.passages]
        self.texts = [p["text"] for p in self.passages]
        tokenized_corpus = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = 3):
        """Return top-k (passage_id, text, score) tuples for a query."""
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.ids, self.texts, scores), key=lambda x: x[2], reverse=True
        )
        return ranked[:k]


if __name__ == "__main__":
    # quick smoke test
    r = BM25Retriever("data/passages.json")
    for pid, text, score in r.retrieve("how much sun do tomatoes need", k=3):
        print(f"{score:.3f}  {pid}  {text[:70]}")
