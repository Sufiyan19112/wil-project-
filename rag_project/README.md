**Group ID:** WIL Project 81
**Team members:** Mohammed Sufiyan(S4191466), 
                  Likith Vudugula(S4203933), 
                  Siva Nelluri(S4158617),
                  Qadar Mohiuddin Syed(S4203189), 
                  Mohammed Faizan Uddin(S4203995).


# Starter RAG Pipeline (Walert-style) — Local Council Services Domain

A minimal Test-Driven RAG pipeline that answers residents' questions about
local council services (bin collection, permits, rates, parking, noise
complaints, etc.) based on a small curated knowledge base. It mirrors the
structure of Walert (RMIT's RAG chatbot) closely enough to count as a
reproduction for Milestone 1, while running entirely on free tools.

**Stakeholder**: residents looking for quick answers instead of digging
through a council website, and council call centre staff who benefit from
fewer repetitive calls about routine questions. Getting this wrong has real
consequences (e.g. wrong bin day, wrong permit advice), so correctly refusing
out-of-scope questions instead of hallucinating an answer is a core value-add,
not just a nice-to-have.

## Structure

```
rag_project/
├── data/
│   ├── passages.json         # council services knowledge base (16 passages)
│   └── test_questions.json   # test set: known / inferred / out_of_kb questions
├── retriever.py               # BM25 retrieval (rank_bm25)
├── generate.py                 # calls a local Ollama model to generate answers
├── eval.py                     # NDCG@k + % correctly-unanswered scoring
├── main.py                     # runs the full pipeline end-to-end
└── requirements.txt
```

## Setup

1. Install Python deps:
   ```
   pip install -r requirements.txt --break-system-packages
   ```

2. Install Ollama (https://ollama.com/) and pull a small model:
   ```
   ollama pull llama3.2
   ```
   Ollama runs a local server at `http://localhost:11434`,  `generate.py`
   talks to it directly, no API key or cost involved.

3. Run the pipeline:
   ```
   python main.py
   ```
   This prints per-question results and a summary, and saves everything to
   `results.json`.

## Extending the knowledge base

Right now there are 16 passages covering common council services (bins,
potholes, parking permits, building/planning permits, rates, parks, tree
removal, noise complaints). To grow it: add more passages to
`data/passages.json`  e.g. real content pulled from your own local council's
public website — each a short, self-contained paragraph with a unique `id`.
Then add matching questions to `test_questions.json` using the same three
types Walert used:

- **known** : Answer comes directly from one passage. Give it
  `"relevant_passages": {"pX": 2}` (label 2 = highly relevant).
- **inferred** : Answer requires combining 2+ passages. Give it
  `"relevant_passages": {"pX": 1, "pY": 1}` (label 1 = partially relevant).
- **out_of_kb** : A plausible question your knowledge base can't answer. Leave
  `"relevant_passages": {}` : The system should refuse rather than hallucinate.

## How this maps to Milestone 1 / the evaluation framework requirement

- **Effectiveness**: `% correctly_unanswered` in the summary output is exactly
  the metric the assessment brief calls out ("% of unanswered questions as in
  Walert"), computed automatically from how many out-of-KB questions get a
  refusal instead of a hallucinated answer.
- **Retrieval quality**: NDCG@k per question type, same measure Walert used.
- **Extending further** (for later milestones): add BERTScore/ROUGE-1 to score
  generated answers against your gold answers, or add a fairness/faithfulness
  check — the brief explicitly says these are welcome additions beyond the
  required effectiveness metric.

## Notes

- `K = 3` in `main.py` matches one of the top k cutoffs Walert tested (they
  tried k=1, 3, 5) worth trying different values and comparing NDCG.
- BM25 here is a stand-in for Walert's BM25 baseline. You could add a dense
  retriever (e.g. sentence-transformers embeddings + cosine similarity) later
  to compare against it, the way Walert compared BM25 vs DPR.
