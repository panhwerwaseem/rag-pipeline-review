# RAG failure modes

How to tell the six failure modes apart using the output of
`scripts/retrieval_eval.py` and `scripts/chunk_stats.py`, and what to try for
each. Every fix listed here is a candidate. Measure it on the same query set
before recommending it.

## Contents

- [Decide in pipeline order](#decide-in-pipeline-order)
- [Stale or incomplete index](#stale-or-incomplete-index)
- [Retrieval miss](#retrieval-miss)
- [Ranking](#ranking)
- [Chunking](#chunking)
- [Context overflow](#context-overflow)
- [Unfaithful generation](#unfaithful-generation)
- [When the label is wrong](#when-the-label-is-wrong)

## Decide in pipeline order

A defect early in the pipeline makes every later stage look broken, so check in
this order and stop at the first stage that fails for the query.

| Question | Evidence | If no |
|---|---|---|
| 1. Is the relevant document in the index, once, in its current version? | `chunk_stats.py --docs`: no `no-chunks`, `duplicate-document`, `exact-duplicate` or `unlisted-document` finding for it | Stale or incomplete index |
| 2. Does the retriever return it at any saved depth? | `retrieval_eval.py`: the relevant ID has a rank | Retrieval miss |
| 3. Is it inside the top k that reaches the prompt? | The rank is at or above k | Ranking |
| 4. Does a retrieved chunk hold the complete passage that answers the question? | Read the retrieved chunk text; `chunk_stats.py` `mid-sentence`, `tiny`, `oversized` | Chunking |
| 5. Did that chunk reach the model whole? | The final prompt as sent, and its token count | Context overflow |
| 6. Does the answer say what the chunk says? | The answer compared claim by claim with the chunks in the prompt | Unfaithful generation |

Questions 1 to 3 are answered by the scripts. Questions 4 to 6 need the chunk
text, the assembled prompt and the answer for the specific query. If those were
not saved, report the stage as not measured.

Document-level labels answer question 2 for the document, not for the passage.
A query can score as retrieved because some chunk of the right document came
back while the chunk that holds the answer did not. Check the chunk text
before you rule out retrieval.

## Stale or incomplete index

**What it looks like.** `retrieval_eval.py` prints `never-retrieved` for an ID
that several queries need. `chunk_stats.py --docs` prints `no-chunks` (a listed
document has no chunk), `unlisted-document` (chunks from a document that is no
longer in the source list), `duplicate-document` or `exact-duplicate` (content
indexed more than once). Answers quote text that was since changed or deleted.

**Confirm.**
- Search the chunk export for the document ID. No hit means it was never
  indexed or was dropped.
- Compare the document list from the source system with the document IDs in
  the index, in both directions.
- For a `near-duplicate` pair, read both chunks. Two versions of the same
  passage with different facts mean an old revision is still indexed.

**Typical causes.**
- A loader failed without raising an error: a scanned PDF with no text layer,
  an unsupported file type, a page behind a login, a size or timeout limit.
- Re-ingestion appends instead of replacing. Chunk IDs are random or include a
  run timestamp, so the same content is stored again on every run.
- The same content is reachable under two paths (a mirror, an archive folder,
  a print view) and both were crawled.
- Deletions in the source are never propagated to the index.

**Candidate fixes.**
- Make ingestion fail loudly, and reconcile source and index counts on every
  run.
- Derive chunk IDs from stable inputs (document ID plus position or a content
  hash) and upsert, so a re-run replaces instead of appending.
- Delete chunks whose document left the source. Exclude mirrors and archives
  at the loader.

Duplicates cost more than storage. Each copy takes a slot in the top k, so
fewer distinct passages reach the prompt, and copies under unlabelled IDs lower
measured precision and rank metrics.

## Retrieval miss

**What it looks like.** The failing query shows `not in the ranking` for a
relevant ID, and the document does have chunks in the export.

**Confirm.**
- Read the chunk that should have matched next to the query. Check whether
  they share vocabulary or only meaning.
- Re-run the query with metadata filters switched off. If the chunk appears,
  the filter or the metadata is wrong.
- Save a deeper ranking. If the chunk appears far down, treat it as a ranking
  problem.

**Typical causes.**
- Vocabulary mismatch. Embedding search can miss exact identifiers, error
  codes, part numbers and rare names. Keyword search misses paraphrases.
- A metadata filter (tenant, language, date, access level) excludes the chunk,
  or the chunk's metadata is missing.
- The query and the documents were embedded differently: a different model or
  model version, or a model that expects a different prefix or input-type
  setting for queries than for documents. Check the embedding model's
  documentation.
- The index compares vectors with a different measure than the model was
  trained for. Unless vectors are normalised to unit length, cosine similarity
  and dot product can rank results differently.
- The query is a follow-up that only makes sense with the conversation
  ("what about the annual plan?") and was sent to retrieval as written.
- The chunk is oversized and the passage was cut off by the embedding model's
  input limit, or is diluted by the rest of the chunk.

**Candidate fixes.** Hybrid search (keyword plus embedding), query rewriting
for follow-ups, fixing filters and metadata, re-embedding with consistent
settings, smaller or structure-aware chunks.

## Ranking

**What it looks like.** The failing query shows `rank N, below k`: the
retriever found the document but too low to reach the prompt. MRR and nDCG are
low while recall at a deeper k is high.

**Confirm.** Score the same results at a larger `--k`. If recall rises
sharply between k and the saved depth, the candidates are there and the order
is the problem. Look at what outranks the relevant chunk: duplicates,
boilerplate and tiny chunks are common.

**Candidate fixes.**
- Remove what is wrongly ranked above it (duplicates, boilerplate) before
  adding anything.
- A reranker over a deeper candidate list. Score the first-stage file and the
  reranked file separately, so you can see whether the reranker gains or loses
  documents.
- A larger k. This costs prompt length and can move the problem to context
  overflow.

## Chunking

**What it looks like.** Retrieval status is ok at document level, yet answers
are incomplete, miss a condition, or stop short. `chunk_stats.py` reports a
high `mid-sentence` share, many `tiny` chunks, or `oversized` chunks. The
passage that answers the question is split across two chunks and only one was
retrieved, or a table row was separated from its header.

**Confirm.** For a failing query, read the retrieved chunks in full. Check
whether any single chunk contains the complete answer. Check the evidence
lines of the `mid-sentence` finding for boundaries inside the passage.

**Typical causes.**
- A fixed-length splitter that ignores sentence, paragraph and heading
  boundaries.
- Chunk size set in one unit and assumed in another (characters against
  tokens).
- Headings stripped or left behind, so a chunk no longer says what it is
  about.
- Tables, lists and code split in the middle.

**Candidate fixes.** Split on structure first (headings, paragraphs,
sentences) and on length second; add overlap; keep the heading path with each
chunk; keep tables and code blocks whole. Use document-level labels to compare
chunkers, because chunk IDs change, and re-run `chunk_stats.py` after the
change.

Document-level retrieval metrics often do not move when chunking gets worse.
That is expected, and it is why chunk shape is checked separately.

## Context overflow

**What it looks like.** The right chunk is in the top k, but the answer
ignores it. The assembled prompt is near or over the model's context limit, or
the application truncates the context to a budget.

**Confirm.**
- Log the final prompt exactly as sent, and check that the chunk is in it and
  complete.
- Count its tokens with the tokenizer of the model in use. Character counts
  are only a rough guide.
- Read the code that assembles the prompt for truncation: which end is cut,
  and whether chat history or the system prompt is allowed to push retrieved
  context out.

**Typical causes.** k raised without a token budget; long chat history;
oversized chunks; truncation from the end, which drops the lowest-ranked chunks
or the question itself.

**Candidate fixes.** A token budget per section of the prompt, enforced before
sending; fewer or smaller chunks; removing duplicates so that k distinct
passages fit. Models can also use material in the middle of a very long
context less reliably than material at the start or end. The size of that
effect depends on the model, so test it with yours before reordering chunks.

## Unfaithful generation

**What it looks like.** The prompt contains a chunk with the correct answer,
complete, and the answer still contradicts it, adds facts that are in no
chunk, or says the information is unavailable.

**Confirm.** List the claims in the answer. Mark each as supported by a chunk
in the prompt, contradicted by one, or absent from all of them. Do this for a
sample of queries with retrieval status ok, and state the sample size.

**Typical causes.**
- The prompt does not tell the model to answer only from the provided
  context, or what to do when the context lacks the answer.
- Two chunks in the prompt disagree, usually an old and a new version of the
  same passage. The answer is faithful to one of them. This is an index
  defect; fix it there.
- The question needs facts from several chunks combined, or a calculation.
- Retrieved text contains instructions and the model followed them.

**Candidate fixes.** Tighten the answer instructions; require citations of
chunk IDs and check them; remove conflicting sources. Measure with a fixed set
of question, context and answer triples that are judged the same way before
and after. This skill's scripts do not score generation.

## When the label is wrong

Before filing a failure, check the label. A query can fail because the
labelled document is not the one that answers it, because a second document
also answers it and was not labelled, or because the corpus changed after
labelling. Fix the query set, record the change, and re-run the baseline. A
baseline from the old labels cannot be compared with a run on the new ones,
and `retrieval_eval.py` refuses it.
