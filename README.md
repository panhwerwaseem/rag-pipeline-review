# rag-pipeline-review

**An agent skill that audits an existing RAG pipeline with measured retrieval quality and chunk checks, instead of guessing at a fix.**

Left alone, a coding agent asked why a RAG system gives bad answers tends to try three or four queries by hand, then recommend a different chunk size, a new embedding model or a reranker. None of that is measured. This skill replaces it with a protocol and two scripts: retrieval is scored separately from generation on a labelled query set, the size of that set is stated with every number, and no change is recommended without a before-and-after comparison on the same queries.

Works with Claude Code, and with any agent that reads the open [Agent Skills](https://agentskills.io) format (Codex, Cursor, GitHub Copilot, Gemini CLI and others).

## What it looks like

The examples run on the fixtures in `skills/rag-pipeline-review/evals/files`, from that folder. The corpus is a small synthetic help centre, and the rankings were saved from a plain keyword (BM25) ranker.

A document that three test questions need was never indexed:

```console
$ python3 ../../scripts/retrieval_eval.py --queries queries.jsonl --results missing-document/results.jsonl --match-field doc_id --k 1,3,5 --show 2
query set: queries.jsonl   24 queries, 24 scored   set id 0b1a8ecfd7a6129b
results:   missing-document/results.jsonl   matched on 'doc_id'   deepest ranking 10

   k   hit rate (95% interval)    recall   precision   MRR     nDCG    n
   1   0.792    (0.595-0.908)      0.750    0.792       0.792   0.792   24
   3   0.917    (0.742-0.977)      0.875    0.542       0.847   0.833   24
   5   0.917    (0.742-0.977)      0.875    0.442       0.847   0.833   24

FAIL  never-retrieved: kb/sso-setup.md is labelled relevant for 3 queries and appears in no ranking in the results file (deepest ranking: 10). Check that it is in the index and that the ID matches.
      q05 (queries.jsonl:5)
      q06 (queries.jsonl:6)
      ... and 1 more (see --json)
failing queries at k=5: 4 of 24 (2 miss, 2 partial)
MISS    q05  "How do I set up SAML single sign-on?"  [queries.jsonl:5]
    relevant   kb/sso-setup.md   not in the ranking (depth 10)
    top 5      kb/password-reset.md#3, kb/two-factor.md#3, kb/rate-limits.md#2, kb/invoices.md#3, kb/webhooks.md#2
MISS    q06  "Are new users created automatically when they sign in through SSO?"  [queries.jsonl:6]
    relevant   kb/sso-setup.md   not in the ranking (depth 10)
    top 5      kb/password-reset.md#3, kb/invoices.md#0, kb/two-factor.md#1, kb/api-keys.md#3, kb/user-roles.md#0
... and 2 more failing (raise --show, or use --json)

RESULT: 1 blocking finding(s), 0 warning(s), n=24 queries
```

The chunk check on the same snapshot says why:

```console
$ python3 ../../scripts/chunk_stats.py --chunks missing-document/chunks.jsonl --docs docs.json
chunks: 44 in 11 document(s)   (missing-document/chunks.jsonl)
length in chars: min 103, p5 118, median 172, mean 175.8, p95 227, max 291
length in words: min 18, p5 20, median 31, mean 30.6, p95 40, max 49
near-duplicate check: ran on 44 distinct chunk texts
ERROR no-chunks (1): documents listed in --docs that have no chunk
      kb/sso-setup.md [docs.json[2]]  listed, but no chunk has this document ID
RESULT: 1 error finding(s), 0 warning finding(s); blocking level: error
```

A splitter that cuts every 300 characters:

```console
$ python3 ../../scripts/chunk_stats.py --chunks mid-sentence/chunks.jsonl --show 3
chunks: 36 in 12 document(s)   (mid-sentence/chunks.jsonl)
length in chars: min 26, p5 63, median 299, mean 238.8, p95 300, max 300
length in words: min 4, p5 9, median 51, mean 41.5, p95 57, max 57
near-duplicate check: ran on 36 distinct chunk texts
ERROR mid-sentence (22): chunks that end without sentence punctuation, 61.1% of 36 non-empty chunks
      kb/refunds.md#0 [mid-sentence/chunks.jsonl:1]  ends ' select the charge you want refunded. Re'; next chunk starts 'funds go back to the orig'
      kb/refunds.md#1 [mid-sentence/chunks.jsonl:2]  ends 'e account holder name and IBAN.\n\nRefunds'; next chunk starts 'are not available for add'
      kb/billing-cycles.md#0 [mid-sentence/chunks.jsonl:4]  ends 'e middle of a cycle, the new plan starts'; next chunk starts 'immediately and we charge'
      ... and 19 more (raise --show, or use --json)
WARN  tiny (1): chunks shorter than 50 characters
      kb/data-export.md#2 [mid-sentence/chunks.jsonl:18]  26 chars: 'ettings in your workspace.'
RESULT: 1 error finding(s), 1 warning finding(s); blocking level: error
```

Before and after on the same query set. Here the "after" index holds every document twice, and the copy outranks the labelled original for every query:

```console
$ python3 ../../scripts/retrieval_eval.py --queries queries.jsonl --results healthy/results.jsonl --match-field doc_id --k 1,3,5 --json > before.json
$ python3 ../../scripts/retrieval_eval.py --queries queries.jsonl --results duplicated-corpus/results.jsonl --match-field doc_id --k 1,3,5 --baseline before.json --show 1
query set: queries.jsonl   24 queries, 24 scored   set id 0b1a8ecfd7a6129b
results:   duplicated-corpus/results.jsonl   matched on 'doc_id'   deepest ranking 10

   k   hit rate (95% interval)    recall   precision   MRR     nDCG    n
   1   0.000    (0.000-0.138)      0.000    0.000       0.000   0.000   24
   3   0.875    (0.690-0.957)      0.833    0.292       0.438   0.532   24
   5   0.958    (0.798-0.993)      0.938    0.292       0.458   0.579   24

failing queries at k=5: 2 of 24 (1 miss, 1 partial)
MISS    q20  "How long before a deleted workspace is really gone?"  [queries.jsonl:20]
    relevant   kb/account-deletion.md   rank 8, below k=5
    top 5      kb-archive/sso-setup.md#2, kb/sso-setup.md#2, kb-archive/rate-limits.md#2, kb/rate-limits.md#2, kb-archive/api-keys.md#1
... and 1 more failing (raise --show, or use --json)

compared with before.json (same query set, n=24, k=5; other k in --json)
  metric          before   after    change
  hit rate@5      1.000    0.958    -0.042
  recall@5        0.979    0.938    -0.042
  precision@5     0.467    0.292    -0.175
  MRR@5           0.927    0.458    -0.469
  nDCG@5          0.929    0.579    -0.351
  per query recall@5: 0 better, 1 worse, 23 same   sign test p=1.0000
  per query MRR@5: 0 better, 24 worse, 0 same   sign test p<0.0001
  per query nDCG@5: 0 better, 24 worse, 0 same   sign test p<0.0001
  A p-value of 0.05 or more means the split of better and worse queries is consistent with chance at this query-set size.
  WORSE   q20: ok -> miss  [queries.jsonl:20]

RESULT: 0 blocking finding(s), 0 warning(s), n=24 queries
```

A baseline from a different query set is refused:

```console
$ head -20 queries.jsonl > fewer.jsonl
$ python3 ../../scripts/retrieval_eval.py --queries fewer.jsonl --results duplicated-corpus/results.jsonl --match-field doc_id --k 1,3,5 --baseline before.json
retrieval_eval.py: error: before.json was computed on a different query set (IDs, texts or labels differ). Re-run the baseline on this query set before comparing.
```

## Install

Pick one route. Installing through more than one gives you the skill twice.

**Claude Code (plugin)**

```
/plugin install rag-pipeline-review --marketplace panhwerwaseem/rag-pipeline-review
```

On Claude Code versions before 2.1.275, use two steps:

```
/plugin marketplace add panhwerwaseem/rag-pipeline-review
/plugin install rag-pipeline-review@panhwerwaseem-rag-pipeline-review
```

**Any agent (skills CLI)**

```bash
npx skills add panhwerwaseem/rag-pipeline-review
```

**Manual**

Copy the `skills/rag-pipeline-review` folder into `.claude/skills/` (Claude Code) or `.agents/skills/` (cross-agent) in your project, or into the same folder under your home directory.

To pin a release, add the tag: `/plugin marketplace add panhwerwaseem/rag-pipeline-review#v0.1.0` or `npx skills add panhwerwaseem/rag-pipeline-review@v0.1.0`. Third-party plugins do not update automatically; run `claude plugin update rag-pipeline-review@panhwerwaseem-rag-pipeline-review` to update.

## Use

Describe the problem and the skill loads on its own:

> Our support bot says it has no information about SSO, but the help article exists. Here is an export of the index and our test questions. What is wrong?

In Claude Code you can also call it directly with `/rag-pipeline-review`.

The scripts need files you export from your pipeline: the chunks in the index, a labelled query set, and the ranked results for those queries. `SKILL.md` gives the formats. The scripts do not run your pipeline and do not call an embedding or language model.

## What is in the skill

| File | Purpose |
|---|---|
| `SKILL.md` | The protocol: six hard rules and an eight-step workflow from mapping the pipeline to a report |
| `scripts/retrieval_eval.py` | Scores saved ranked results against a labelled query set. Reports hit rate, recall, precision, MRR and nDCG at each k with the query count, lists every failing query with what was retrieved instead, flags labelled IDs that appear in no ranking, and compares against a saved baseline from the same query set |
| `scripts/chunk_stats.py` | Reads an export of the chunks. Reports the length distribution, empty, tiny and oversized chunks, duplicate IDs, exact and near-duplicate chunks, boilerplate repeated across documents, duplicated documents, chunks cut mid-sentence, documents with no chunks, and chunks from documents that are not in the document list |
| `references/failure-modes.md` | How to tell apart an incomplete index, a retrieval miss, a ranking problem, bad chunking, context overflow and unfaithful generation, with candidate fixes for each |
| `references/review-checklist.md` | What to check in ingestion, chunking, embedding and index, retrieval, generation and operations, and how to verify each point |
| `references/query-set.md` | How to build and label a query set, how many queries are enough, and the exact metric definitions |
| `evals/` | Four saved snapshots of a small synthetic pipeline (healthy, duplicated corpus, mid-sentence chunking, a document missing from the index) with eval cases, plus trigger queries |

### How the numbers are computed

For one query, with R the labelled relevant IDs and the ranking cut at k:

- hit rate@k: 1 if any of the top k results is in R, otherwise 0
- recall@k: labelled IDs found in the top k, divided by the size of R
- precision@k: top-k results that are in R, divided by k
- MRR@k: 1 divided by the rank of the first result in R, or 0 if there is none in the top k
- nDCG@k: DCG@k divided by the best possible DCG@k, where DCG@k is the sum of grade / log2(rank + 1), and the grade is 1 for ungraded labels

Each reported value is the mean over the scored queries. The interval next to the hit rate is the 95% Wilson score interval. A comparison counts the queries that got better and worse and applies a two-sided sign test. Near-duplicate chunks are found by exact Jaccard similarity over 3-word shingles. `references/query-set.md` has the details, including how repeated IDs are counted.

### What the scripts do not do

- They do not score generated answers. Faithfulness and answer quality need saved prompts and answers and are covered by the checklist, not by a script.
- They do not embed text, query a vector store or call a model.
- The mid-sentence check is a punctuation heuristic for prose. It is not meant for code or transcripts; `--no-sentence-check` turns it off.
- Boilerplate is detected only when the repeated text is a chunk of its own, not when it is embedded in otherwise unique chunks.

## Requirements

- Python 3.9 or later
- No packages to install and no network access
- Input files as JSON or JSONL. YAML is read only if PyYAML is already installed

## What it reads, writes and contacts

- **Reads:** the query, results, chunk, document-list and baseline files you name on the command line.
- **Writes:** nothing. Reports go to standard output; redirect them if you want a file.
- **Runs:** no other program.
- **Contacts:** nothing. The scripts contain no network code; CI enforces this.
- **Permissions:** the skill pre-approves no tools. Your agent asks before running commands, as usual.

Script output quotes query text and short excerpts of chunk text. When an agent runs the scripts, that output becomes part of the agent's context. See [SECURITY.md](SECURITY.md).

## Testing status

- The scripts have unit tests that run in CI on Python 3.9 and 3.12, together with a validator for the skill format and plugin manifests. The metric formulas are tested against values worked out by hand, and the near-duplicate search is tested against a brute-force comparison.
- The eval cases in `skills/rag-pipeline-review/evals/` are in the [skill-creator](https://github.com/anthropics/skills/tree/main/skills/skill-creator) format. A with-skill versus without-skill score has not been published yet; results are welcome as an issue or pull request.

## Development

```bash
python3 tools/validate.py              # skill format, manifests, no-network check
python3 -m unittest discover -s tests  # script unit tests
```

## License

[MIT](LICENSE)
