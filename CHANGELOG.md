# Changelog

All notable changes to this skill are recorded here. Versions follow [Semantic Versioning](https://semver.org).

## 0.1.0 - 2026-11-06

### Added

- `SKILL.md` with the review protocol: hard rules and an eight-step workflow.
- `scripts/retrieval_eval.py` for hit rate, recall, precision, MRR and nDCG at k on a labelled query set, a list of failing queries, and baseline comparison on the same query set.
- `scripts/chunk_stats.py` for chunk length distribution, empty, tiny, oversized, duplicate, near-duplicate, boilerplate and mid-sentence chunks, and documents with no chunks.
- References for failure modes, a review checklist, and building a labelled query set.
- Four saved pipeline snapshots (one healthy, three with seeded defects) with eval cases, and trigger queries.
- Unit tests, a format validator, and a CI workflow.
