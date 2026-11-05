# Review checklist

What to check at each stage of an existing RAG pipeline, and how to verify
each point. "Script" marks checks that `scripts/chunk_stats.py` or
`scripts/retrieval_eval.py` performs. Everything else needs the code, the
configuration, or a saved prompt and answer. Report a point you could not
verify as not checked. Do not mark it as passed.

Limits, defaults and parameter names differ between embedding models, vector
stores and frameworks, and between their versions. Read them from the
project's configuration and the vendor's documentation. Do not assume them.

## Contents

- [Ingestion](#ingestion)
- [Chunking](#chunking)
- [Embedding and index](#embedding-and-index)
- [Retrieval](#retrieval)
- [Generation](#generation)
- [Operations](#operations)

## Ingestion

| Check | How to verify |
|---|---|
| Every source document is indexed | Script: `chunk_stats.py --docs` reports `no-chunks`. Build the list from the source system, not from the index |
| Nothing is indexed that left the source | Script: `unlisted-document` |
| Nothing is indexed twice | Script: `exact-duplicate`, `duplicate-document`, `duplicate-id` |
| Loader failures are visible | Read the ingestion code for swallowed exceptions and for files skipped by type or size. Look for a per-run count of documents read, chunks written and failures |
| Extracted text is the real content | Read a sample of chunks from each source type. Look for navigation menus, cookie banners, page headers and footers, and text in the wrong reading order. Script: `boilerplate` finds blocks repeated across documents |
| Scanned or image-only files are handled | Find one in the source and search the export for its ID. A missing or empty chunk means there is no text layer and no OCR step |
| Tables and code survive extraction | Read the chunks for a document that has them |
| Re-ingestion replaces, not appends | Check how chunk IDs are made. Stable IDs with upsert replace; random IDs or timestamps append |
| Metadata needed for filtering is set on every chunk | Count chunks that lack each filter field in the export |

## Chunking

| Check | How to verify |
|---|---|
| Chunks end on sentence or paragraph boundaries | Script: `mid-sentence` share |
| No empty, tiny or oversized chunks | Script: `empty`, `tiny`, `oversized`. Set `--max-chars` from the embedding model's input limit |
| Chunk size is in the unit the code assumes | Compare the configured size with the length distribution the script prints. For English text a token is usually several characters, so a size meant as tokens and applied as characters gives chunks several times smaller than intended |
| Each chunk can be understood alone | Read a sample. Check whether a chunk says which document, section or product it is about, or depends on a heading that was left behind |
| Overlap is intended | With overlap, neighbouring chunks share text. Check that the overlap setting matches what the export shows |
| One passage that answers a question fits in one chunk | For failing queries, read the retrieved chunks |

## Embedding and index

| Check | How to verify |
|---|---|
| Queries and documents use the same embedding model and version | Compare the ingestion and the query code paths. A model upgrade requires re-embedding the whole index |
| Query and document settings follow the model's documentation | Some models expect a different prefix or input type for queries than for documents |
| Text longer than the model's input limit is not cut silently | Depending on the model and client library, over-long input is rejected or truncated. Find out which, and compare the limit with the longest chunks |
| The index's distance measure matches the model | Compare the index configuration with the model's documentation, and check whether vectors are normalised |
| Approximate search is not losing results | If the index uses approximate nearest-neighbour search, compare its results for the query set with an exact search and score both files |
| The index holds what the export says | Compare the chunk count in the index with the export |

## Retrieval

| Check | How to verify |
|---|---|
| Retrieval quality is measured, with n stated | Script: `retrieval_eval.py` on a labelled set. See `query-set.md` |
| The measured k is the k the prompt uses | Read the retrieval call and the prompt assembly. Score at that k |
| Filters do not exclude relevant chunks | Re-run failing queries without filters and score both files |
| Follow-up questions are made self-contained before retrieval | Read how the query is built from the conversation. Include follow-up questions in the query set |
| The same result is not returned twice | Script: `repeated-result`. Common when keyword and embedding results are merged without removing duplicates |
| A reranker earns its place | Score the first-stage file and the reranked file against the same queries, and compare with `--baseline` |
| Score thresholds are justified | If results under a similarity score are dropped, find where the number came from. Score with and without it |
| Questions with no answer in the corpus are handled | Keep such queries in the set with an empty `relevant` list and check what the system returns for them |

## Generation

The scripts do not score generation. These points need saved prompts and
answers.

| Check | How to verify |
|---|---|
| The prompt limits the answer to the provided context | Read the prompt template. Check that it says what to do when the context does not contain the answer |
| Retrieved chunks reach the model whole | Log the final prompt for a few queries. Count tokens with the model's tokenizer. Read the truncation code |
| Answers are supported by the retrieved chunks | For a sample of queries with retrieval status ok, mark each claim as supported, contradicted or absent. State the sample size |
| Sources are cited and the citations are real | Check that cited chunk IDs were in the prompt and say what the answer claims |
| Retrieved text is treated as data | Check how chunks are delimited in the prompt. Documents can contain text written as instructions; try one |
| Access control is applied at retrieval | Check that a user's query can only retrieve chunks that user may read. Filtering after generation is too late |

## Operations

| Check | How to verify |
|---|---|
| The index is refreshed when sources change | Find the schedule or trigger. Compare the newest source change with the newest indexed change |
| Deletions propagate | Delete or rename a test document in the source and re-run ingestion. Script: `unlisted-document` |
| The labelled query set is versioned and re-run on change | Look for it in the repository and in CI. `retrieval_eval.py --min` gives a pass or fail exit code |
| Retrieval is logged per request | Check that the query, retrieved chunk IDs, scores and the index or model version are recorded, so a bad answer can be traced to a stage |
| Failed or empty retrievals are counted | Look for a metric or log line for zero results and for answers that decline |
| Changing the embedding model has a plan | Vectors from different models cannot be mixed in one index. Check for a rebuild and switch-over procedure |
| Cost and latency are known per stage | Ask for, or measure, the time and cost of embedding the query, searching, reranking and generating. Report them as not measured if there is no data |
