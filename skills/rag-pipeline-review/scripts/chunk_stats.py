#!/usr/bin/env python3
"""Report chunk-level defects in an exported RAG index.

Input: a file of chunks, JSONL (one JSON object per line) or a JSON array.
Each record needs an ID, a document ID and the chunk text:
  {"id": "handbook.md#3", "doc_id": "handbook.md", "text": "..."}
Other key names are set with --id-field, --doc-field and --text-field; a dotted
path such as metadata.source reads a nested key. Records must be in document
order, because the mid-sentence check looks at the next chunk of a document.
YAML is read only if PyYAML is installed.

Checks (E = error, W = warning):
  E empty               text is empty or whitespace
  E duplicate-id        two records share a chunk ID
  E exact-duplicate     same text after collapsing whitespace and case
  E boilerplate         same text in --boilerplate-docs or more documents whose
                        content otherwise differs (headers, footers, templates)
  E duplicate-document  documents whose sets of chunk texts are identical
  E no-chunks           a document listed in --docs has no chunk
  W unlisted-document   chunks whose document is not listed in --docs
  W near-duplicate      Jaccard similarity of 3-word shingles at or above
                        --near-dup-threshold (exact search, not sampled)
  W tiny / oversized    shorter than --min-chars; longer than --max-chars or
                        than --oversize-factor times the median length
  W mid-sentence        the last line does not end in sentence punctuation and
                        is either 6 or more words long or followed, in the same
                        document, by a chunk that starts in lower case. Becomes
                        an error when the share reaches --max-cut-share.
The mid-sentence check is a prose heuristic. Headings, list items, table rows
and indented or fenced code are skipped. For code, transcripts or other text
without sentence punctuation, pass --no-sentence-check.

Examples:
  chunk_stats.py --chunks chunks.jsonl
  chunk_stats.py --chunks chunks.jsonl --docs docs.json --max-chars 2000
  chunk_stats.py --chunks export.jsonl --text-field page_content --doc-field metadata.source --json

Exit codes: 0 = ran, nothing at the blocking level. 1 = at least one finding
at the blocking level (errors by default; --fail-on warn includes warnings).
2 = usage or input error.
Standard library only. Reads the files you name, writes to stdout. No network.
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

TERMINAL = ".!?:;\u2026\u3002\uff01\uff1f"  # also ellipsis and CJK full stop, ! and ?
CLOSERS = "\"'\u201d\u2019)]}*_`\u00bb"  # closing quotes, brackets, Markdown emphasis
OPENERS = "\"'\u201c\u2018([*_\u00ab"
MARKER_RE = re.compile(r"^(\s*(#{1,6}\s|[-*+\u2022]\s|[>|]|\d+[.)]\s|```|~~~)|\t| {4})")
PROSE_WORDS = 6
SHINGLE_WORDS = 3
YAML_HELP = ("%s is YAML and PyYAML is not installed. Either install it (pip install pyyaml), "
             "or convert the file to JSON or JSONL first (for example with yq: "
             "yq -o=json . file.yaml), or export the data from your pipeline as JSONL.")


class InputError(Exception):
    """A bad input file or option. Printed as one line; exit code 2."""


def load_records(path):
    """Return [(location, value)] from a JSON array, a JSONL file, or a YAML list."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")  # also accepts a byte-order mark
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError("cannot read %s: %s" % (path, exc))
    if str(path).lower().endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            raise InputError(YAML_HELP % path)
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise InputError("%s is not valid YAML: %s" % (path, exc))
    elif text.lstrip().startswith("["):
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise InputError("%s is not valid JSON: %s" % (path, exc))
    else:
        records = []
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(("%s:%d" % (path, number), json.loads(line)))
                except ValueError as exc:
                    raise InputError("%s:%d is not valid JSON (%s). Expected a JSON array, "
                                     "or one JSON object per line." % (path, number, exc))
        return records
    if not isinstance(data, list):
        raise InputError("%s must contain a list of records" % path)
    return [("%s[%d]" % (path, index), value) for index, value in enumerate(data)]


def field(record, dotted, where, kind):
    value = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise InputError("%s: no %r key for the %s (set --%s-field)" % (where, dotted, kind, kind))
        value = value[part]
    return value


def parse_chunks(records, id_field, doc_field, text_field):
    chunks = []
    for where, rec in records:
        if not isinstance(rec, dict):
            raise InputError("%s: a chunk record must be a JSON object" % where)
        text = field(rec, text_field, where, "text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise InputError("%s: %r must be a string" % (where, text_field))
        chunks.append({"where": where, "id": str(field(rec, id_field, where, "id")),
                       "doc": str(field(rec, doc_field, where, "doc")), "text": text})
    if not chunks:
        raise InputError("the chunk file has no records")
    return chunks


def parse_doc_ids(records):
    ids = []
    for where, value in records:
        if isinstance(value, dict):
            if "id" not in value:
                raise InputError('%s: a document record needs an "id"' % where)
            value = value["id"]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise InputError("%s: a document ID must be a string or an integer" % where)
        ids.append((str(value), where))
    return ids


def normalise(text):
    return " ".join(text.split()).casefold()


def percentile(sorted_values, pct):
    """Nearest-rank percentile: the value at position ceil(pct / 100 * n), counting from 1."""
    return sorted_values[max(1, int(math.ceil(pct / 100.0 * len(sorted_values)))) - 1]


def distribution(values):
    ordered = sorted(values)
    return {"min": ordered[0], "p5": percentile(ordered, 5), "median": statistics.median(ordered),
            "mean": round(sum(ordered) / float(len(ordered)), 1),
            "p95": percentile(ordered, 95), "max": ordered[-1]}


def shingles(normalised, size=SHINGLE_WORDS):
    words = re.findall(r"\w+", normalised)
    if len(words) < size:
        return set([" ".join(words)]) if words else set()
    return set(" ".join(words[i:i + size]) for i in range(len(words) - size + 1))


def near_duplicates(texts, threshold):
    """Every pair (i, j, jaccard) of texts with shingle Jaccard similarity >= threshold.

    Exact. Uses prefix filtering: with shingles ordered rarest first, two sets
    that reach the threshold must share a shingle within the first
    len(set) - ceil(threshold * len(set)) + 1 shingles of each, so only those
    are indexed to find candidate pairs. Every candidate is then checked in full.
    """
    sets = [shingles(text) for text in texts]
    frequency = {}
    for group in sets:
        for shingle in group:
            frequency[shingle] = frequency.get(shingle, 0) + 1
    index, candidates = {}, set()
    for i, group in enumerate(sets):
        if not group:
            continue
        ordered = sorted(group, key=lambda s: (frequency[s], s))
        prefix = len(ordered) - int(math.ceil(threshold * len(ordered) - 1e-9)) + 1
        for shingle in ordered[:prefix]:
            for j in index.get(shingle, ()):
                candidates.add((j, i))
            index.setdefault(shingle, []).append(i)
    pairs = []
    for i, j in candidates:
        shared = len(sets[i] & sets[j])
        jaccard = shared / float(len(sets[i]) + len(sets[j]) - shared)
        if jaccard >= threshold - 1e-12:
            pairs.append((i, j, jaccard))
    return sorted(pairs, key=lambda pair: (-pair[2], pair[0], pair[1]))


def last_line(text):
    lines = [line for line in text.rstrip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def ends_open(text):
    """True when the last line is not a heading, list item, table row or code,
    and does not end in sentence punctuation (closing quotes and brackets ignored)."""
    line = last_line(text)
    if not line or MARKER_RE.match(line):
        return False
    tail = line.rstrip().rstrip(CLOSERS).rstrip()
    return bool(tail) and tail[-1] not in TERMINAL


def starts_lower(text):
    head = text.lstrip().lstrip(OPENERS)
    return bool(head) and head[0].islower()


def analyse(chunks, doc_ids, args):
    findings = []

    def add(level, kind, summary, items):
        if items:
            findings.append({"level": level, "kind": kind, "count": len(items),
                             "summary": summary, "items": items})

    def item(chunk, detail):
        return {"where": chunk["where"], "id": chunk["id"], "doc": chunk["doc"], "detail": detail}

    total = len(chunks)
    full = [c for c in chunks if c["text"].strip()]
    add("error", "empty", "chunks with no text",
        [item(c, "empty or whitespace only") for c in chunks if not c["text"].strip()])

    by_id = {}
    for chunk in chunks:
        by_id.setdefault(chunk["id"], []).append(chunk)
    add("error", "duplicate-id", "chunk IDs used by more than one record",
        [item(group[0], "also at " + ", ".join(c["where"] for c in group[1:]))
         for _, group in sorted(by_id.items()) if len(group) > 1])

    lengths = [len(c["text"].strip()) for c in full]
    stats = {"chunks": total, "documents": len(set(c["doc"] for c in chunks))}
    if full:
        stats["chars"] = distribution(lengths)
        stats["words"] = distribution([len(c["text"].split()) for c in full])
        median = stats["chars"]["median"]
        add("warn", "tiny", "chunks shorter than %d characters" % args.min_chars,
            [item(c, "%d chars: %r" % (n, c["text"].strip()[:60]))
             for c, n in zip(full, lengths) if n < args.min_chars])
        big = []
        for chunk, n in zip(full, lengths):
            if args.max_chars and n > args.max_chars:
                big.append(item(chunk, "%d chars, above --max-chars %d" % (n, args.max_chars)))
            elif n > args.oversize_factor * median:
                big.append(item(chunk, "%d chars, more than %g times the median of %g"
                                % (n, args.oversize_factor, median)))
        add("warn", "oversized", "chunks far longer than the rest or above the set limit", big)

    # Exact duplicates, boilerplate and duplicated documents.
    by_text, doc_texts, doc_first = {}, {}, {}
    for chunk in full:
        norm = normalise(chunk["text"])
        by_text.setdefault(norm, []).append(chunk)
        doc_texts.setdefault(chunk["doc"], set()).add(norm)
        doc_first.setdefault(chunk["doc"], chunk)
    by_content = {}
    for doc, texts in doc_texts.items():
        by_content.setdefault(frozenset(texts), []).append(doc)
    content_of = dict((doc, key) for key, docs in by_content.items() for doc in docs)
    add("error", "duplicate-document", "documents whose chunk texts are identical to another document's",
        [{"where": doc_first[docs[0]]["where"], "id": "", "doc": docs[0],
          "detail": "same content as " + ", ".join(
              "%s [%s]" % (doc, doc_first[doc]["where"]) for doc in docs[1:])}
         for docs in sorted(sorted(group) for group in by_content.values()) if len(docs) > 1])
    duplicates, boilerplate, redundant = [], [], 0
    for norm, group in by_text.items():
        if len(group) < 2:
            continue
        docs = sorted(set(c["doc"] for c in group))
        detail = "%d copies in %d document(s): %s  %r" % (
            len(group), len(docs), ", ".join("%s [%s]" % (c["id"], c["where"]) for c in group[1:]),
            group[0]["text"].strip()[:50])
        if len(set(content_of[doc] for doc in docs)) >= args.boilerplate_docs:
            boilerplate.append(item(group[0], detail))
        else:
            duplicates.append(item(group[0], detail))
            redundant += len(group) - 1
    add("error", "exact-duplicate", "groups of identical chunks; %d redundant chunk(s), %.1f%% of all"
        % (redundant, 100.0 * redundant / total), duplicates)
    add("error", "boilerplate", "texts repeated across %d or more documents with otherwise "
        "different content" % args.boilerplate_docs, boilerplate)

    firsts = [group[0] for group in by_text.values()]
    if len(firsts) > args.near_dup_limit:
        stats["near_duplicate_check"] = ("skipped: %d distinct chunk texts is above "
                                         "--near-dup-limit %d" % (len(firsts), args.near_dup_limit))
    else:
        stats["near_duplicate_check"] = "ran on %d distinct chunk texts" % len(firsts)
        pairs = near_duplicates([normalise(c["text"]) for c in firsts], args.near_dup_threshold)
        add("warn", "near-duplicate", "pairs of chunks with shingle Jaccard similarity of %g or more"
            % args.near_dup_threshold,
            [item(firsts[i], "%.2f similar to %s [%s]" % (score, firsts[j]["id"], firsts[j]["where"]))
             for i, j, score in pairs])

    if not args.no_sentence_check and full:
        following = {}
        previous = {}
        for chunk in full:
            if chunk["doc"] in previous:
                following[id(previous[chunk["doc"]])] = chunk
            previous[chunk["doc"]] = chunk
        cuts = []
        for chunk in full:
            if not ends_open(chunk["text"]):
                continue
            nxt = following.get(id(chunk))
            continues = bool(nxt and starts_lower(nxt["text"]))
            if len(last_line(chunk["text"]).split()) >= PROSE_WORDS or continues:
                cuts.append(item(chunk, "ends %r%s" % (
                    chunk["text"].rstrip()[-40:],
                    "; next chunk starts %r" % nxt["text"].lstrip()[:25] if continues else "")))
        share = len(cuts) / float(len(full))
        stats["mid_sentence_share"] = round(share, 4)
        add("error" if share >= args.max_cut_share else "warn", "mid-sentence",
            "chunks that end without sentence punctuation, %.1f%% of %d non-empty chunks"
            % (100 * share, len(full)), cuts)

    if doc_ids is not None:
        listed = set(doc for doc, _ in doc_ids)
        present = set(c["doc"] for c in chunks)
        add("error", "no-chunks", "documents listed in --docs that have no chunk",
            [{"where": where, "id": "", "doc": doc, "detail": "listed, but no chunk has this document ID"}
             for doc, where in doc_ids if doc not in present])
        strays = {}
        for chunk in chunks:
            if chunk["doc"] not in listed:
                strays.setdefault(chunk["doc"], []).append(chunk)
        add("warn", "unlisted-document", "documents with chunks that are not listed in --docs",
            [item(group[0], "%d chunk(s) from a document that is not in --docs" % len(group))
             for _, group in sorted(strays.items())])

    order = {"error": 0, "warn": 1}
    findings.sort(key=lambda f: order[f["level"]])
    return stats, findings


def render(path, stats, findings, show, fail_on):
    out = ["chunks: %d in %d document(s)   (%s)" % (stats["chunks"], stats["documents"], path)]
    for unit in ("chars", "words"):
        if unit in stats:
            d = stats[unit]
            out.append("length in %s: min %g, p5 %g, median %g, mean %g, p95 %g, max %g" % (
                unit, d["min"], d["p5"], d["median"], d["mean"], d["p95"], d["max"]))
    out.append("near-duplicate check: %s" % stats["near_duplicate_check"])
    for finding in findings:
        out.append("%-5s %s (%d): %s" % (finding["level"].upper(), finding["kind"],
                                         finding["count"], finding["summary"]))
        for entry in finding["items"][:show]:
            label = entry["id"] or entry["doc"]
            out.append("      %s%s  %s" % (label, " [%s]" % entry["where"] if entry["where"] else "",
                                           entry["detail"]))
        if finding["count"] > show:
            out.append("      ... and %d more (raise --show, or use --json)" % (finding["count"] - show))
    errors = sum(1 for f in findings if f["level"] == "error")
    out.append("RESULT: %d error finding(s), %d warning finding(s); blocking level: %s"
               % (errors, len(findings) - errors, fail_on))
    return "\n".join(out)


def emit(text):
    """Print, escaping any character the output encoding cannot represent."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, "backslashreplace").decode(encoding))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", required=True, help="chunk export (JSONL or JSON array)")
    parser.add_argument("--docs", help="list of every document ID that should be indexed "
                        "(JSON array or JSONL of IDs, or of objects with an \"id\")")
    parser.add_argument("--id-field", default="id", help="key holding the chunk ID (default id)")
    parser.add_argument("--doc-field", default="doc_id", help="key holding the document ID (default doc_id)")
    parser.add_argument("--text-field", default="text", help="key holding the chunk text (default text)")
    parser.add_argument("--min-chars", type=int, default=50, help="tiny below this (default 50)")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="oversized above this; set it from your embedding model's input limit")
    parser.add_argument("--oversize-factor", type=float, default=3.0,
                        help="oversized above this many times the median length (default 3)")
    parser.add_argument("--near-dup-threshold", type=float, default=0.8,
                        help="Jaccard similarity that counts as a near-duplicate (default 0.8)")
    parser.add_argument("--near-dup-limit", type=int, default=50000,
                        help="skip the near-duplicate check above this many distinct texts (default 50000)")
    parser.add_argument("--boilerplate-docs", type=int, default=3,
                        help="documents sharing a text before it counts as boilerplate (default 3)")
    parser.add_argument("--max-cut-share", type=float, default=0.2,
                        help="share of mid-sentence chunks that makes the finding an error (default 0.2)")
    parser.add_argument("--no-sentence-check", action="store_true", help="skip the mid-sentence check")
    parser.add_argument("--fail-on", choices=("error", "warn"), default="error",
                        help="lowest level that gives exit code 1 (default error)")
    parser.add_argument("--show", type=int, default=5, help="items to print per finding (default 5)")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)
    if not 0 < args.near_dup_threshold <= 1:
        parser.error("--near-dup-threshold must be above 0 and at most 1")
    if args.boilerplate_docs < 2:
        parser.error("--boilerplate-docs must be 2 or more")
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        chunks = parse_chunks(load_records(args.chunks), args.id_field, args.doc_field, args.text_field)
        doc_ids = parse_doc_ids(load_records(args.docs)) if args.docs else None
    except InputError as exc:
        print("chunk_stats.py: error: %s" % exc, file=sys.stderr)
        return 2
    stats, findings = analyse(chunks, doc_ids, args)
    if args.json:
        emit(json.dumps({"source": args.chunks, "stats": stats, "findings": findings},
                        indent=2, sort_keys=True))
    else:
        emit(render(args.chunks, stats, findings, args.show, args.fail_on))
    blocking = ("error",) if args.fail_on == "error" else ("error", "warn")
    return 1 if any(f["level"] in blocking for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
