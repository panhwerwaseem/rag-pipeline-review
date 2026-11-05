---
name: rag-pipeline-review
description: Audits an existing retrieval-augmented generation (RAG) pipeline by scoring saved retrieval results against a labelled query set (recall@k, precision@k, MRR, nDCG, hit rate) and checking exported chunks for duplicates, mid-sentence cuts, empty or oversized chunks and documents that were never indexed. Use when a RAG chatbot or document search gives wrong or incomplete answers or says it does not know although the source exists, when answers got worse after re-indexing, when someone proposes changing chunk size, embedding model, top-k, hybrid search or a reranker, before a RAG feature ships, or when the user says review our RAG, evaluate retrieval, retrieval quality, chunking strategy, recall at k, or vector search returns irrelevant results.
license: MIT
compatibility: Requires Python 3.9 or later to run the bundled scripts. No network access and no extra packages needed.
metadata:
  author: panhwerwaseem
  version: "0.1.0"
---

# RAG pipeline review

A review is done when four things are true: retrieval has been measured on its
own against a labelled query set of stated size; the chunk export has been
checked; every failing query has been assigned to a failure mode with evidence;
and every recommended change either carries a before-and-after measurement on
the same query set or is labelled an untested hypothesis. This skill audits a
pipeline that already exists. It does not design or scaffold a new one.

Paths in this file are relative to this skill's directory.

## Hard rules

1. **Measure retrieval separately from generation.** Do not blame the model,
   the prompt or the retriever for a bad answer until
   `scripts/retrieval_eval.py` output shows whether the relevant chunks were in
   the top k for that query. Right chunks and a wrong answer is a generation
   problem. Missing chunks is a retrieval or index problem. Do not infer one
   from the other.
2. **No change without a before and an after.** Do not recommend changing
   chunk size, overlap, embedding model, k, hybrid search or a reranker as a
   fix unless you can show `retrieval_eval.py --baseline` output for that
   change on the same query set. Without it, present the change as a
   hypothesis and name the measurement that would test it.
3. **No verdict from hand-picked queries.** Trying a few queries is not a
   measurement. Do not call retrieval good, bad, improved or regressed on fewer
   than 20 labelled queries, or while the script prints `small-query-set`.
   Build or extend the set first (`references/query-set.md`), and do not write
   labels only from what the pipeline returned.
4. **State n with every number.** Every metric you report carries the query-set
   size, k and match level, for example "recall@5 0.71 (n=48, document
   level)". Copy numbers from script output produced in this session. Never
   estimate a metric from reading code or configuration.
5. **Do not call models or paid APIs unasked.** The scripts only read saved
   files. If producing the results file means running the user's pipeline
   (embedding or model calls, cost, data leaving the machine), ask first and
   say what will be sent where.
6. **Audit, do not rebuild.** Do not swap the vector store or framework, and
   do not add components, as part of a review. Report each finding with the
   smallest change that addresses it and how to verify that change.

## Workflow

### 1. Map the pipeline

Read the code and configuration and write down: sources and loaders, the
splitter and its settings (size, overlap, unit), the embedding model and its
input limit, the index and its distance metric, k, metadata filters, hybrid
search or reranking, how the prompt is assembled, and the generation model.
Note where chunks and ranked results can be exported.

### 2. Collect the inputs

Put them in a scratch folder such as `.rag-review/` and keep it out of version
control. Each file is JSONL or a JSON array.

| File | One record | Notes |
|---|---|---|
| `chunks.jsonl` | `{"id": "...", "doc_id": "...", "text": "..."}` | Every chunk in the index, in document order. Other key names work through `--id-field`, `--doc-field`, `--text-field` |
| `docs.json` | a list of document IDs | Every document that should be indexed, taken from the source system and not from the index |
| `queries.jsonl` | `{"id": "q1", "query": "...", "relevant": ["doc-7"]}` | The labelled query set. If there is none, build one with `references/query-set.md` before step 4 |
| `results.jsonl` | `{"id": "q1", "results": [{"id": "...", "doc_id": "..."}, ...]}` | The pipeline's ranked results for those queries, best first, saved at least twice as deep as k |

If the pipeline has a reranker, save two results files: the first-stage
candidates and the reranked list.

### 3. Check the chunks

```bash
python3 scripts/chunk_stats.py --chunks .rag-review/chunks.jsonl --docs .rag-review/docs.json
```

Add `--max-chars N` from the embedding model's input limit. Errors (empty
chunks, duplicate IDs, exact duplicates, boilerplate, duplicated documents,
documents with no chunks, a high share of mid-sentence cuts) are defects to
report with the listed evidence. Warnings (tiny, oversized, near-duplicate,
unlisted documents, a few mid-sentence cuts) need a look at the listed chunks
before you report them. For code or transcripts add `--no-sentence-check`.

### 4. Score retrieval

```bash
python3 scripts/retrieval_eval.py --queries .rag-review/queries.jsonl --results .rag-review/results.jsonl --k 1,3,5 --match-field doc_id
```

Make the largest k the number of chunks the pipeline puts in the prompt. Use
`--match-field doc_id` when labels are document IDs and results are chunks;
drop it when labels are chunk IDs. Then save the report for later comparison:

```bash
python3 scripts/retrieval_eval.py --queries .rag-review/queries.jsonl --results .rag-review/results.jsonl --k 1,3,5 --match-field doc_id --json > .rag-review/before.json
```

Exit code 1 means a blocking finding: a labelled ID that appears in no ranking,
a labelled query with no results, or a `--min` threshold not met.

### 5. Assign every failing query to a failure mode

| Script output for the query | Failure mode | Next check |
|---|---|---|
| `never-retrieved`, and `chunk_stats.py` reports `no-chunks` for that document | Incomplete index | Why ingestion dropped the document |
| `not in the ranking`, but the document has chunks | Retrieval miss | Filters, query wording against chunk wording, embedding settings |
| `rank N, below k` | Ranking | Whether a deeper k, hybrid search or a reranker moves it up, measured |
| Status ok, answer still wrong or incomplete | Chunking, context overflow or generation | Read the retrieved chunks and the final prompt for that query |

`references/failure-modes.md` gives the evidence that separates each mode and
the candidate fixes. Work from the start of the pipeline: an index defect makes
every later stage look broken.

### 6. Review what the scripts cannot see

Walk `references/review-checklist.md` for ingestion, retrieval, generation and
operations. For generation, take queries whose retrieval status is ok and
compare each claim in the saved answer with the retrieved chunks. If no
answers were saved, report generation as not measured.

### 7. Test a change before recommending it

Apply the change in a copy or branch, re-export the results for the same
queries, and compare:

```bash
python3 scripts/retrieval_eval.py --queries .rag-review/queries.jsonl --results .rag-review/results-after.jsonl --k 1,3,5 --match-field doc_id --baseline .rag-review/before.json
```

The script refuses a baseline made from a different query set, k or match
field. Read the sign-test line: with p at 0.05 or above, report "no measurable
difference at n=N", not an improvement. Re-run `chunk_stats.py` too when the
change touches ingestion or chunking.

### 8. Report

```
Pipeline:        <loader -> splitter (size, overlap) -> embedding model -> index -> k -> reranker -> model>
Query set:       <n> labelled queries, <how they were collected and labelled>, <document|chunk> level
Retrieval:       hit rate@<k> <x> (95% interval <lo>-<hi>), recall@<k> <x>, MRR@<k> <x>, nDCG@<k> <x>  (n=<n>)
Chunks:          <chunks> in <documents> documents; errors <kind: count>; warnings <kind: count>
Failing queries: <count> of <n>: <a> index, <b> retrieval miss, <c> ranking, <d> chunking/context/generation
Findings:        <one per line: failure mode, evidence (query ID, chunk ID, file:line), effect>
Recommended:     <change>: measured <metric before -> after, n, sign test p>
                 <change>: untested hypothesis; test by <measurement>
Not measured:    <generation faithfulness, latency, cost, anything skipped, and why>
```

## Edge cases

- **No labelled query set and no time to build one.** Run `chunk_stats.py`
  only, report retrieval quality as not measured, and make no claim about it.
- **`no-overlap` finding.** The labels and results use different IDs. Fix
  `--match-field` or the export. Do not report the zeros.
- **Document-level labels.** They survive re-chunking, so they are the right
  choice for comparing chunkers. They cannot show whether the right passage was
  retrieved or whether a chunk was cut mid-sentence. Say so, and rely on
  `chunk_stats.py` for chunk shape.
- **Chunk-level labels and a chunking change.** Old chunk IDs no longer exist,
  so the comparison is invalid. Relabel, or switch to document-level labels.
- **Reranker.** Score the first-stage file and the reranked file against the
  same queries. A document present in the first and missing from the second
  was lost by the reranker or its cut-off.
- **Questions the corpus cannot answer.** Keep them in the set with an empty
  `relevant` list. They are left out of the averages. Check separately that
  the system declines to answer them.
- **Very large index.** A sample is enough for length and mid-sentence checks.
  Duplicate, boilerplate and `no-chunks` findings need the full export; say
  which you ran.
- **Private data.** The scratch folder holds corpus text and user queries. Do
  not commit it or paste it into third-party tools.
