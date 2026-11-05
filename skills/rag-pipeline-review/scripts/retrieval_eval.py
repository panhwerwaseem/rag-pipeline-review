#!/usr/bin/env python3
"""Score saved retrieval results against a labelled query set.

Inputs are files you already have. Each is a JSON array or JSONL (one JSON
object per line). YAML is read only if PyYAML is installed.

  --queries  one record per query:
               {"id": "q1", "query": "how do refunds work", "relevant": ["doc-7"]}
             "relevant" is a list of IDs, or an object of ID -> grade (a number
             above 0) for graded relevance. A query with no relevant ID is
             reported and left out of the averages.
  --results  one record per query, best result first:
               {"id": "q1", "results": [{"id": "doc-7#2", "doc_id": "doc-7"}, ...]}
             A result is an ID or an object. --match-field names the key that
             is compared with the labels (default "id"; use "doc_id" when the
             labels are document IDs and the results are chunks).

Metrics, each the mean over the scored queries, for every k in --k:
  hit rate   1 if any of the top k results is relevant, else 0
  recall     labelled IDs found in the top k / labelled IDs
  precision  top-k results that are relevant / k (k, even if fewer came back)
  MRR        1 / rank of the first relevant result, 0 if none in the top k
  nDCG       DCG / ideal DCG, where DCG is the sum of grade / log2(rank + 1)
             over the top k and an ungraded label has grade 1
If a labelled ID occurs more than once in one ranking, recall and nDCG count
it once, at its first rank. A labelled query with no results record scores 0.

Examples:
  retrieval_eval.py --queries queries.jsonl --results results.jsonl --k 1,3,5
  retrieval_eval.py --queries q.jsonl --results r.jsonl --match-field doc_id --min recall@5=0.8
  retrieval_eval.py --queries q.jsonl --results before.jsonl --json > before.json
  retrieval_eval.py --queries q.jsonl --results after.jsonl --baseline before.json

Exit codes: 0 = ran, nothing blocking. 1 = a blocking finding: a --min
threshold not met, a labelled query with no results record, or a labelled ID
that appears in no ranking in the results file. 2 = usage or input error.
Standard library only. Reads the files you name, writes to stdout. No network.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

METRICS = ("hit_rate", "recall", "precision", "mrr", "ndcg")
TITLES = {"hit_rate": "hit rate", "recall": "recall", "precision": "precision",
          "mrr": "MRR", "ndcg": "nDCG"}
STATUS_ORDER = {"ok": 0, "partial": 1, "miss": 2}
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


def as_id(value, where):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise InputError("%s: an ID must be a string or an integer, got %r" % (where, value))
    return str(value)


def parse_queries(records):
    """Return [{"id", "query", "where", "grades": {relevant id: grade}}]."""
    queries, seen = [], set()
    for where, rec in records:
        if not isinstance(rec, dict) or "id" not in rec or "relevant" not in rec:
            raise InputError('%s: a query record needs "id" and "relevant"' % where)
        qid = as_id(rec["id"], where)
        if qid in seen:
            raise InputError("%s: duplicate query id %r" % (where, qid))
        seen.add(qid)
        relevant, grades = rec["relevant"], {}
        if isinstance(relevant, list):
            grades = dict((as_id(value, where), 1.0) for value in relevant)
        elif isinstance(relevant, dict):
            for key, grade in relevant.items():
                if isinstance(grade, bool) or not isinstance(grade, (int, float)) or grade < 0:
                    raise InputError("%s: grade for %r must be a number, 0 or above" % (where, key))
                if grade > 0:
                    grades[str(key)] = float(grade)
        else:
            raise InputError('%s: "relevant" must be a list of IDs or an object of ID -> grade' % where)
        queries.append({"id": qid, "query": str(rec.get("query", "")), "where": where,
                        "grades": grades})
    if not queries:
        raise InputError("the query file has no records")
    return queries


def parse_results(records, match_field):
    """Return {query id: {"where", "ranked": [(matched id, shown id)]}} in rank order."""
    results = {}
    for where, rec in records:
        if not isinstance(rec, dict) or "id" not in rec or not isinstance(rec.get("results"), list):
            raise InputError('%s: a results record needs "id" and a "results" list' % where)
        qid = as_id(rec["id"], where)
        if qid in results:
            raise InputError("%s: duplicate results for query id %r" % (where, qid))
        ranked = []
        for position, item in enumerate(rec["results"], 1):
            if isinstance(item, dict):
                if match_field not in item:
                    raise InputError("%s: result %d has no %r key (see --match-field)"
                                     % (where, position, match_field))
                matched = as_id(item[match_field], where)
                shown = as_id(item["id"], where) if "id" in item else matched
            else:
                matched = shown = as_id(item, where)
            ranked.append((matched, shown))
        results[qid] = {"where": where, "ranked": ranked}
    return results


def dcg(gains):
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))


def score_query(ranked_ids, grades, k):
    """Metrics at cutoff k for one query. grades maps relevant ID -> grade above 0."""
    top = ranked_ids[:k]
    found, gains, first_rank = set(), [], None
    for rank, rid in enumerate(top, 1):
        if rid in grades and first_rank is None:
            first_rank = rank
        if rid in grades and rid not in found:
            found.add(rid)
            gains.append(grades[rid])
        else:
            gains.append(0.0)
    ideal = sorted(grades.values(), reverse=True)[:k]
    return {
        "hit_rate": 1.0 if found else 0.0,
        "recall": len(found) / float(len(grades)),
        "precision": sum(1 for rid in top if rid in grades) / float(k),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": dcg(gains) / dcg(ideal),
    }


def wilson_interval(successes, n):
    """95% Wilson score interval for a proportion."""
    z = 1.959964
    p = successes / float(n)
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (centre - half) / (1 + z * z / n)), min(1.0, (centre + half) / (1 + z * z / n))


def sign_test_p(wins, losses):
    """Two-sided exact sign test: chance of a split at least this uneven if wins and
    losses were equally likely. Ties are left out by the caller."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / float(2 ** n)
    return min(1.0, 2 * tail)


def fingerprint(queries):
    """Hash of the query IDs, texts and labels, so two runs can prove they used the same set."""
    canon = sorted([q["id"], q["query"], sorted(q["grades"].items())] for q in queries)
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def evaluate(queries, results, ks):
    cutoff = max(ks)
    per_query = []
    for query in queries:
        grades = query["grades"]
        if not grades:
            continue
        entry = results.get(query["id"])
        ranked = entry["ranked"] if entry else []
        ids = [matched for matched, _ in ranked]
        scores = dict((str(k), score_query(ids, grades, k)) for k in ks)
        ranks = {}
        for rank, rid in enumerate(ids, 1):
            ranks.setdefault(rid, rank)
        at = scores[str(cutoff)]
        status = "miss" if not at["hit_rate"] else ("partial" if at["recall"] < 1 else "ok")
        per_query.append({
            "id": query["id"], "query": query["query"], "where": query["where"],
            "status": status, "depth": len(ranked), "has_results": entry is not None,
            "relevant": [{"id": rid, "rank": ranks.get(rid)} for rid in sorted(grades)],
            "retrieved": [shown for _, shown in ranked[:cutoff]],
            "scores": scores,
        })
    n = len(per_query)
    metrics = {}
    for k in ks:
        row = dict((m, round(sum(q["scores"][str(k)][m] for q in per_query) / n, 6)) for m in METRICS)
        hits = sum(1 for q in per_query if q["scores"][str(k)]["hit_rate"])
        row["hit_rate_ci95"] = [round(v, 4) for v in wilson_interval(hits, n)]
        metrics[str(k)] = row
    return per_query, metrics


def build_findings(queries, results, per_query, metrics, ks, thresholds, min_queries):
    findings = []

    def add(level, kind, message, evidence):
        findings.append({"level": level, "kind": kind, "message": message, "evidence": evidence})

    n = len(per_query)
    for metric, k, floor in thresholds:
        value = metrics[str(k)][metric]
        if value < floor:
            low = sorted((q["scores"][str(k)][metric], q["id"], q["where"]) for q in per_query
                         if q["scores"][str(k)][metric] < floor)
            add("fail", "threshold", "%s@%d is %.3f, below the required %.3f (n=%d); "
                "%d quer%s below it" % (TITLES[metric], k, value, floor, n, len(low),
                                        "y scores" if len(low) == 1 else "ies score"),
                ["%s scores %.3f (%s)" % (qid, score, where) for score, qid, where in low])
    absent = [q for q in per_query if not q["has_results"]]
    if absent:
        add("fail", "missing-results",
            "%d labelled quer%s no record in the results file; scored as zero"
            % (len(absent), "y has" if len(absent) == 1 else "ies have"),
            ["%s (%s)" % (q["id"], q["where"]) for q in absent])
    everywhere = set(matched for entry in results.values() for matched, _ in entry["ranked"])
    wanted = {}
    for query in queries:
        for rid in query["grades"]:
            if rid not in everywhere:
                wanted.setdefault(rid, []).append("%s (%s)" % (query["id"], query["where"]))
    deepest = max([len(entry["ranked"]) for entry in results.values()] or [0])
    labelled = set(rid for query in queries for rid in query["grades"])
    if len(wanted) == len(labelled):
        add("fail", "no-overlap",
            "none of the %d labelled IDs appears anywhere in the results file. The labels and the "
            "results probably use different kinds of ID; check --match-field." % len(labelled),
            sorted(wanted)[:5])
        wanted = {}
    for rid in sorted(wanted):
        add("fail", "never-retrieved",
            "%s is labelled relevant for %d quer%s and appears in no ranking in the results file "
            "(deepest ranking: %d). Check that it is in the index and that the ID matches."
            % (rid, len(wanted[rid]), "y" if len(wanted[rid]) == 1 else "ies", deepest), wanted[rid])
    if n < min_queries:
        add("warn", "small-query-set",
            "only %d scored quer%s (--min-queries %d). Treat these numbers as anecdotes, "
            "not as a measurement." % (n, "y" if n == 1 else "ies", min_queries), [])
    unlabelled = [q for q in queries if not q["grades"]]
    if unlabelled:
        add("warn", "unlabelled-queries",
            "%d quer%s no relevant ID and %s left out of the averages"
            % (len(unlabelled), "y has" if len(unlabelled) == 1 else "ies have",
               "is" if len(unlabelled) == 1 else "are"),
            ["%s (%s)" % (q["id"], q["where"]) for q in unlabelled])
    repeats = []
    for qid in sorted(results):
        shown = [s for _, s in results[qid]["ranked"]]
        twice = sorted(set(s for s in shown if shown.count(s) > 1))
        if twice:
            repeats.append("%s (%s): %s" % (qid, results[qid]["where"], ", ".join(twice)))
    if repeats:
        add("warn", "repeated-result",
            "%d ranking(s) return the same result more than once" % len(repeats), repeats)
    if max(ks) > deepest:
        add("warn", "shallow-results",
            "k=%d is deeper than any ranking in the results file (deepest: %d)" % (max(ks), deepest), [])
    return findings


def compare(report, baseline, path):
    """Deltas against an earlier --json report. Refuses a different query set."""
    same = baseline["query_set"]["fingerprint"] == report["query_set"]["fingerprint"]
    old_rows = dict((q["id"], q) for q in baseline["per_query"])
    old_metrics, old_cutoff = baseline["metrics"], baseline["cutoff"]
    old_field = baseline["match_field"]
    if not same:
        raise InputError("%s was computed on a different query set (IDs, texts or labels differ). "
                         "Re-run the baseline on this query set before comparing." % path)
    if old_cutoff != report["cutoff"] or old_field != report["match_field"]:
        raise InputError("%s used k=%s and --match-field %s; this run uses k=%s and %s. "
                         "Compare like with like." % (path, old_cutoff, old_field,
                                                      report["cutoff"], report["match_field"]))
    cut = str(report["cutoff"])
    deltas = {}
    for k in sorted(set(old_metrics) & set(report["metrics"]), key=int):
        deltas[k] = dict((m, {"before": old_metrics[k][m], "after": report["metrics"][k][m],
                              "change": round(report["metrics"][k][m] - old_metrics[k][m], 6)})
                         for m in METRICS)
    paired = {}
    for metric in ("recall", "mrr", "ndcg"):
        better = worse = 0
        for q in report["per_query"]:
            diff = q["scores"][cut][metric] - old_rows[q["id"]]["scores"][cut][metric]
            better += diff > 1e-9
            worse += diff < -1e-9
        paired[metric] = {"better": better, "worse": worse,
                          "same": len(report["per_query"]) - better - worse,
                          "sign_test_p": round(sign_test_p(better, worse), 6)}
    changed = []
    for q in report["per_query"]:
        before = old_rows[q["id"]]["status"]
        if before != q["status"]:
            changed.append({"id": q["id"], "where": q["where"], "before": before, "after": q["status"],
                            "worse": STATUS_ORDER[q["status"]] > STATUS_ORDER[before]})
    return {"baseline": str(path), "deltas": deltas, "paired": paired, "status_changes": changed}


def render(report, show):
    out = []
    qs, cut, n = report["query_set"], report["cutoff"], report["query_set"]["scored"]
    out.append("query set: %s   %d queries, %d scored   set id %s"
               % (qs["path"], qs["queries"], n, qs["fingerprint"]))
    out.append("results:   %s   matched on %r   deepest ranking %d"
               % (report["results"]["path"], report["match_field"], report["results"]["deepest"]))
    out.append("")
    out.append("   k   hit rate (95% interval)    recall   precision   MRR     nDCG    n")
    for k in report["k"]:
        row = report["metrics"][str(k)]
        out.append("%4d   %.3f    (%.3f-%.3f)      %.3f    %.3f       %.3f   %.3f   %d" % (
            k, row["hit_rate"], row["hit_rate_ci95"][0], row["hit_rate_ci95"][1],
            row["recall"], row["precision"], row["mrr"], row["ndcg"], n))
    out.append("")
    for finding in report["findings"]:
        out.append("%s  %s: %s" % (finding["level"].upper(), finding["kind"], finding["message"]))
        for line in finding["evidence"][:show]:
            out.append("      " + line)
        if len(finding["evidence"]) > show:
            out.append("      ... and %d more (see --json)" % (len(finding["evidence"]) - show))
    failing = [q for q in report["per_query"] if q["status"] != "ok"]
    misses = sum(1 for q in failing if q["status"] == "miss")
    out.append("failing queries at k=%d: %d of %d (%d miss, %d partial)"
               % (cut, len(failing), n, misses, len(failing) - misses))
    for q in failing[:show]:
        out.append("%-7s %s  %s  [%s]" % (q["status"].upper(), q["id"],
                                          json.dumps(q["query"], ensure_ascii=False), q["where"]))
        for rel in q["relevant"]:
            if rel["rank"] is None:
                place = "not in the ranking (depth %d)" % q["depth"]
            elif rel["rank"] > cut:
                place = "rank %d, below k=%d" % (rel["rank"], cut)
            else:
                place = "rank %d" % rel["rank"]
            out.append("    relevant   %s   %s" % (rel["id"], place))
        out.append("    top %-6d %s" % (cut, ", ".join(q["retrieved"]) or "(no results)"))
    if len(failing) > show:
        out.append("... and %d more failing (raise --show, or use --json)" % (len(failing) - show))
    comparison = report.get("comparison")
    if comparison:
        out.append("")
        out.append("compared with %s (same query set, n=%d, k=%d; other k in --json)"
                   % (comparison["baseline"], n, cut))
        out.append("  metric          before   after    change")
        for metric in METRICS:
            d = comparison["deltas"][str(cut)][metric]
            out.append("  %-15s %.3f    %.3f    %+.3f" % (
                "%s@%d" % (TITLES[metric], cut), d["before"], d["after"], d["change"]))
        for metric in ("recall", "mrr", "ndcg"):
            p = comparison["paired"][metric]
            shown = "<0.0001" if p["sign_test_p"] < 0.0001 else "=%.4f" % p["sign_test_p"]
            out.append("  per query %s@%d: %d better, %d worse, %d same   sign test p%s" % (
                TITLES[metric], cut, p["better"], p["worse"], p["same"], shown))
        out.append("  A p-value of 0.05 or more means the split of better and worse queries is "
                   "consistent with chance at this query-set size.")
        for change in comparison["status_changes"][:show]:
            out.append("  %s  %s: %s -> %s  [%s]" % ("WORSE " if change["worse"] else "BETTER",
                       change["id"], change["before"], change["after"], change["where"]))
        if len(comparison["status_changes"]) > show:
            out.append("  ... and %d more status changes (see --json)"
                       % (len(comparison["status_changes"]) - show))
    blocking = sum(1 for f in report["findings"] if f["level"] == "fail")
    out.append("")
    out.append("RESULT: %d blocking finding(s), %d warning(s), n=%d queries"
               % (blocking, len(report["findings"]) - blocking, n))
    return "\n".join(out)


def emit(text):
    """Print, escaping any character the output encoding cannot represent."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, "backslashreplace").decode(encoding))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", required=True, help="labelled query set (JSON array or JSONL)")
    parser.add_argument("--results", required=True, help="ranked results per query (JSON array or JSONL)")
    parser.add_argument("--k", default="5", help="cutoffs, comma separated (default 5). Failing "
                        "queries and comparisons are judged at the largest")
    parser.add_argument("--match-field", default="id",
                        help="key of a result object compared with the labels (default id)")
    parser.add_argument("--min", action="append", default=[], metavar="METRIC@K=VALUE",
                        help="fail if a mean is below VALUE, e.g. recall@5=0.8. Metrics: %s. "
                        "Repeatable" % ", ".join(METRICS))
    parser.add_argument("--min-queries", type=int, default=20,
                        help="warn when fewer queries than this are scored (default 20)")
    parser.add_argument("--baseline", help="an earlier --json report on the same query set to compare with")
    parser.add_argument("--show", type=int, default=20,
                        help="failing queries and evidence lines to print (default 20)")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)
    try:
        args.ks = sorted(set(int(part) for part in args.k.split(",")))
    except ValueError:
        parser.error("--k must be integers separated by commas, e.g. 1,3,5")
    if not args.ks or args.ks[0] < 1:
        parser.error("--k values must be 1 or more")
    args.thresholds = []
    for spec in args.min:
        try:
            name, value = spec.split("=")
            metric, k = name.split("@")
            threshold = (metric, int(k), float(value))
        except ValueError:
            parser.error("--min %r is not METRIC@K=VALUE, e.g. recall@5=0.8" % spec)
        if metric not in METRICS or threshold[1] not in args.ks:
            parser.error("--min %r: the metric must be one of %s and K must be listed in --k"
                         % (spec, ", ".join(METRICS)))
        args.thresholds.append(threshold)
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    try:
        queries = parse_queries(load_records(args.queries))
        results = parse_results(load_records(args.results), args.match_field)
        if not any(q["grades"] for q in queries):
            raise InputError("no query in %s has a relevant ID; nothing can be scored" % args.queries)
        per_query, metrics = evaluate(queries, results, args.ks)
        report = {
            "query_set": {"path": args.queries, "queries": len(queries), "scored": len(per_query),
                          "fingerprint": fingerprint(queries)},
            "results": {"path": args.results,
                        "deepest": max([len(e["ranked"]) for e in results.values()] or [0])},
            "match_field": args.match_field, "k": args.ks, "cutoff": max(args.ks),
            "metrics": metrics,
            "findings": build_findings(queries, results, per_query, metrics, args.ks,
                                       args.thresholds, args.min_queries),
            "per_query": per_query,
        }
        if args.baseline:
            try:
                baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise InputError("cannot read baseline %s: %s" % (args.baseline, exc))
            try:
                report["comparison"] = compare(report, baseline, args.baseline)
            except (KeyError, TypeError):
                raise InputError("%s is not a retrieval_eval.py --json report" % args.baseline)
    except InputError as exc:
        print("retrieval_eval.py: error: %s" % exc, file=sys.stderr)
        return 2
    emit(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report, args.show))
    return 1 if any(f["level"] == "fail" for f in report["findings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
