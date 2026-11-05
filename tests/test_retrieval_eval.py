import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "rag-pipeline-review"
FILES = SKILL / "evals" / "files"
sys.path.insert(0, str(SKILL / "scripts"))

import retrieval_eval  # noqa: E402


def run(argv):
    """Run the script's main(); return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = retrieval_eval.main(argv)
        except SystemExit as exc:  # argparse usage errors
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def run_fixture(name, *extra):
    code, out, _ = run(["--queries", str(FILES / "queries.jsonl"),
                        "--results", str(FILES / name / "results.jsonl"),
                        "--match-field", "doc_id", "--k", "1,3,5", "--json"] + list(extra))
    return code, json.loads(out)


def kinds(report, level=None):
    return [f["kind"] for f in report["findings"] if level is None or f["level"] == level]


class MetricFormulaTests(unittest.TestCase):
    """Every expected value below was worked out by hand, not by the script."""

    def test_binary_relevance_at_k5(self):
        # Relevant: a, b, c. Ranking: x a y b z. Hits at ranks 2 and 4.
        s = retrieval_eval.score_query(["x", "a", "y", "b", "z"], {"a": 1.0, "b": 1.0, "c": 1.0}, 5)
        self.assertEqual(s["hit_rate"], 1.0)
        self.assertAlmostEqual(s["recall"], 2 / 3.0)
        self.assertAlmostEqual(s["precision"], 2 / 5.0)
        self.assertAlmostEqual(s["mrr"], 1 / 2.0)
        # DCG  = 1/log2(3) + 1/log2(5)  = 0.63093 + 0.43068 = 1.06161
        # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63093 + 0.5 = 2.13093
        self.assertAlmostEqual(s["ndcg"], 0.49819, places=5)

    def test_binary_relevance_at_k3_and_k1(self):
        grades = {"a": 1.0, "b": 1.0, "c": 1.0}
        s3 = retrieval_eval.score_query(["x", "a", "y", "b", "z"], grades, 3)
        self.assertAlmostEqual(s3["recall"], 1 / 3.0)
        self.assertAlmostEqual(s3["precision"], 1 / 3.0)
        self.assertAlmostEqual(s3["mrr"], 0.5)
        # DCG = 0.63093, IDCG = 2.13093 (three relevant IDs fit in the top 3)
        self.assertAlmostEqual(s3["ndcg"], 0.29608, places=5)
        s1 = retrieval_eval.score_query(["x", "a", "y", "b", "z"], grades, 1)
        self.assertEqual([s1[m] for m in retrieval_eval.METRICS], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_perfect_ranking_scores_one_and_precision_divides_by_k(self):
        s = retrieval_eval.score_query(["a", "b", "c"], {"a": 1.0, "b": 1.0, "c": 1.0}, 5)
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["mrr"], 1.0)
        self.assertAlmostEqual(s["ndcg"], 1.0)
        self.assertAlmostEqual(s["precision"], 3 / 5.0)  # k, not the 3 results returned

    def test_ideal_dcg_is_cut_at_k(self):
        # Three relevant IDs but k=1: the best possible top-1 is one relevant result.
        s = retrieval_eval.score_query(["a"], {"a": 1.0, "b": 1.0, "c": 1.0}, 1)
        self.assertAlmostEqual(s["ndcg"], 1.0)
        self.assertAlmostEqual(s["recall"], 1 / 3.0)

    def test_graded_relevance(self):
        # Grades a=3, b=1. Ranking: b a.
        # DCG  = 1/log2(2) + 3/log2(3) = 1 + 1.89279 = 2.89279
        # IDCG = 3/log2(2) + 1/log2(3) = 3 + 0.63093 = 3.63093
        s = retrieval_eval.score_query(["b", "a"], {"a": 3.0, "b": 1.0}, 2)
        self.assertAlmostEqual(s["ndcg"], 0.79671, places=5)
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["mrr"], 1.0)

    def test_reciprocal_rank_is_zero_when_first_hit_is_below_k(self):
        self.assertEqual(retrieval_eval.score_query(["x", "y", "a"], {"a": 1.0}, 2)["mrr"], 0.0)
        self.assertAlmostEqual(retrieval_eval.score_query(["x", "y", "a"], {"a": 1.0}, 3)["mrr"], 1 / 3.0)

    def test_repeated_labelled_id_counts_once_for_recall_and_ndcg(self):
        # Ranking a a b with a, b relevant (two chunks of document a).
        s = retrieval_eval.score_query(["a", "a", "b"], {"a": 1.0, "b": 1.0}, 3)
        self.assertEqual(s["recall"], 1.0)
        self.assertEqual(s["precision"], 1.0)  # each result is judged on its own
        # Gains 1, 0, 1: DCG = 1 + 1/log2(4) = 1.5. IDCG = 1 + 0.63093 = 1.63093
        self.assertAlmostEqual(s["ndcg"], 0.91972, places=5)
        only_repeats = retrieval_eval.score_query(["a", "a", "a"], {"a": 1.0}, 3)
        self.assertAlmostEqual(only_repeats["ndcg"], 1.0)

    def test_empty_ranking_scores_zero(self):
        s = retrieval_eval.score_query([], {"a": 1.0}, 5)
        self.assertEqual([s[m] for m in retrieval_eval.METRICS], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_dcg(self):
        self.assertAlmostEqual(retrieval_eval.dcg([1, 0, 1]), 1.5)
        self.assertEqual(retrieval_eval.dcg([]), 0)


class StatisticsTests(unittest.TestCase):
    def test_wilson_interval_for_five_of_ten(self):
        low, high = retrieval_eval.wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=4)
        self.assertAlmostEqual(high, 0.7634, places=4)

    def test_wilson_interval_at_the_extremes(self):
        # With p = 0 the interval is 0 to z^2 / (n + z^2) = 3.8415 / 27.8415.
        low, high = retrieval_eval.wilson_interval(0, 24)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.1380, places=4)
        low, high = retrieval_eval.wilson_interval(24, 24)
        self.assertAlmostEqual(low, 0.8620, places=4)
        self.assertAlmostEqual(high, 1.0)
        self.assertLessEqual(high, 1.0)

    def test_sign_test(self):
        self.assertAlmostEqual(retrieval_eval.sign_test_p(5, 0), 2 / 32.0)
        self.assertAlmostEqual(retrieval_eval.sign_test_p(8, 1), 2 * 10 / 512.0)
        self.assertAlmostEqual(retrieval_eval.sign_test_p(1, 8), 2 * 10 / 512.0)
        self.assertEqual(retrieval_eval.sign_test_p(3, 3), 1.0)
        self.assertEqual(retrieval_eval.sign_test_p(0, 0), 1.0)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name, rows, as_array=False):
        path = self.dir / name
        if as_array:
            path.write_text(json.dumps(rows))
        else:
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return str(path)

    def report(self, queries, results, *extra, **kwargs):
        code, out, err = run(["--queries", self.write("q.jsonl", queries, kwargs.get("as_array", False)),
                              "--results", self.write("r.jsonl", results, kwargs.get("as_array", False)),
                              "--json"] + list(extra))
        return code, (json.loads(out) if out else None), err

    def test_means_are_taken_over_scored_queries(self):
        code, report, _ = self.report(
            [{"id": "q1", "query": "one", "relevant": ["a"]},
             {"id": "q2", "query": "two", "relevant": ["b", "c"]}],
            [{"id": "q1", "results": ["a", "x", "b", "c"]}, {"id": "q2", "results": ["x", "y", "b", "c"]}],
            "--k", "2")
        self.assertEqual(code, 0)
        row = report["metrics"]["2"]
        self.assertEqual((row["hit_rate"], row["recall"], row["precision"], row["mrr"]),
                         (0.5, 0.5, 0.25, 0.5))
        self.assertEqual(report["query_set"]["scored"], 2)
        self.assertEqual([q["status"] for q in report["per_query"]], ["ok", "miss"])

    def test_json_array_input_and_integer_ids(self):
        code, report, _ = self.report(
            [{"id": 1, "relevant": [7]}], [{"id": 1, "results": [{"id": 7, "score": 0.9}]}],
            "--k", "1", as_array=True)
        self.assertEqual(code, 0)
        self.assertEqual(report["metrics"]["1"]["recall"], 1.0)
        self.assertTrue(report["per_query"][0]["where"].endswith("q.jsonl[0]"))

    def test_small_query_set_is_a_warning_not_a_failure(self):
        code, report, _ = self.report([{"id": "q1", "relevant": ["a"]}], [{"id": "q1", "results": ["a"]}])
        self.assertEqual(code, 0)
        self.assertIn("small-query-set", kinds(report, "warn"))

    def test_labelled_query_without_results_is_scored_zero_and_blocks(self):
        code, report, _ = self.report(
            [{"id": "q1", "relevant": ["a"]}, {"id": "q2", "relevant": ["a"]}],
            [{"id": "q1", "results": ["a"]}], "--k", "1")
        self.assertEqual(code, 1)
        self.assertIn("missing-results", kinds(report, "fail"))
        self.assertEqual(report["metrics"]["1"]["recall"], 0.5)

    def test_query_without_labels_is_left_out_and_reported(self):
        code, report, _ = self.report(
            [{"id": "q1", "relevant": ["a"]}, {"id": "q2", "relevant": []}],
            [{"id": "q1", "results": ["a"]}, {"id": "q2", "results": ["a"]}], "--k", "1")
        self.assertEqual(code, 0)
        self.assertEqual(report["query_set"]["scored"], 1)
        self.assertIn("unlabelled-queries", kinds(report, "warn"))

    def test_grade_zero_means_not_relevant(self):
        code, report, _ = self.report(
            [{"id": "q1", "relevant": {"a": 2, "b": 0}}], [{"id": "q1", "results": ["b", "a"]}], "--k", "2")
        self.assertEqual(report["metrics"]["2"]["mrr"], 0.5)
        self.assertEqual([r["id"] for r in report["per_query"][0]["relevant"]], ["a"])

    def test_same_result_twice_and_k_deeper_than_rankings_are_warnings(self):
        code, report, _ = self.report(
            [{"id": "q1", "relevant": ["a"]}], [{"id": "q1", "results": ["a", "a"]}], "--k", "5")
        self.assertEqual(code, 0)
        self.assertIn("repeated-result", kinds(report, "warn"))
        self.assertIn("shallow-results", kinds(report, "warn"))

    def test_bad_inputs_exit_two_with_a_message_and_no_traceback(self):
        good_q = [{"id": "q1", "relevant": ["a"]}]
        good_r = [{"id": "q1", "results": ["a"]}]
        cases = [
            ([{"id": "q1"}], good_r, '"relevant"'),
            (good_q + good_q, good_r, "duplicate query id"),
            ([{"id": "q1", "relevant": "a"}], good_r, "must be a list"),
            ([{"id": "q1", "relevant": {"a": -1}}], good_r, "0 or above"),
            (good_q, [{"id": "q1", "results": "a"}], '"results" list'),
            (good_q, good_r + good_r, "duplicate results"),
            (good_q, [{"id": "q1", "results": [{"chunk": "a"}]}], "--match-field"),
            ([{"id": "q1", "relevant": []}], good_r, "nothing can be scored"),
        ]
        for queries, results, expected in cases:
            code, report, err = self.report(queries, results)
            self.assertEqual(code, 2, expected)
            self.assertIsNone(report)
            self.assertIn(expected, err)
            self.assertNotIn("Traceback", err)

    def test_broken_jsonl_names_the_line(self):
        path = self.dir / "broken.jsonl"
        path.write_text('{"id": "q1", "relevant": ["a"]}\n{not json}\n')
        code, _, err = run(["--queries", str(path), "--results", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("broken.jsonl:2", err)

    def test_missing_file_and_bad_options_exit_two(self):
        self.assertEqual(run(["--queries", str(self.dir / "none.jsonl"), "--results", "x"])[0], 2)
        path = self.write("q.jsonl", [{"id": "q1", "relevant": ["a"]}])
        for extra in (["--k", "0"], ["--k", "five"], ["--min", "recall=0.8"],
                      ["--min", "recall@3=0.8"], ["--min", "speed@5=1"]):
            self.assertEqual(run(["--queries", path, "--results", path] + extra)[0], 2, extra)

    def test_byte_order_mark_and_non_ascii_text_on_an_ascii_terminal(self):
        queries = self.dir / "q.jsonl"
        queries.write_bytes(b"\xef\xbb\xbf" + json.dumps(
            {"id": "q1", "query": "Wie \u00e4ndere ich das Passwort?", "relevant": ["a", "b"]},
            ensure_ascii=False).encode("utf-8") + b"\n")
        results = self.write("r.jsonl", [{"id": "q1", "results": ["a", "b", "x"]}])
        raw = io.BytesIO()
        terminal = io.TextIOWrapper(raw, encoding="ascii")
        with redirect_stdout(terminal):
            code = retrieval_eval.main(["--queries", str(queries), "--results", results, "--k", "1"])
            terminal.flush()
        self.assertEqual(code, 0)
        self.assertIn(b"Wie \\xe4ndere ich das Passwort?", raw.getvalue())

    def test_yaml_without_pyyaml_exits_two_and_says_how_to_convert(self):
        path = self.dir / "queries.yaml"
        path.write_text("- id: q1\n  relevant: [a]\n")
        saved = sys.modules.get("yaml", "absent")
        sys.modules["yaml"] = None  # makes "import yaml" raise ImportError
        try:
            code, _, err = run(["--queries", str(path), "--results", str(path)])
        finally:
            if saved == "absent":
                del sys.modules["yaml"]
            else:
                sys.modules["yaml"] = saved
        self.assertEqual(code, 2)
        self.assertIn("PyYAML is not installed", err)
        self.assertIn("JSON", err)


class FixtureTests(unittest.TestCase):
    def test_healthy_snapshot_has_no_blocking_finding(self):
        code, report = run_fixture("healthy")
        self.assertEqual(code, 0)
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["query_set"]["scored"], 24)
        self.assertEqual(report["metrics"]["5"]["hit_rate"], 1.0)
        # 23 queries find every labelled document in the top 5; q24 finds one of two.
        self.assertAlmostEqual(report["metrics"]["5"]["recall"], 23.5 / 24, places=6)
        failing = [q for q in report["per_query"] if q["status"] != "ok"]
        self.assertEqual([(q["id"], q["status"]) for q in failing], [("q24", "partial")])
        self.assertIn({"id": "kb/data-export.md", "rank": 10}, failing[0]["relevant"])

    def test_missing_document_is_reported_with_the_queries_that_need_it(self):
        code, report = run_fixture("missing-document")
        self.assertEqual(code, 1)
        self.assertEqual(kinds(report, "fail"), ["never-retrieved"])
        finding = report["findings"][0]
        self.assertIn("kb/sso-setup.md", finding["message"])
        self.assertEqual([line.split()[0] for line in finding["evidence"]], ["q05", "q06", "q07"])
        self.assertTrue(all("queries.jsonl:" in line for line in finding["evidence"]))
        status = dict((q["id"], q["status"]) for q in report["per_query"])
        self.assertEqual((status["q05"], status["q06"], status["q07"]), ("miss", "miss", "partial"))
        q05 = [q for q in report["per_query"] if q["id"] == "q05"][0]
        self.assertEqual(q05["relevant"], [{"id": "kb/sso-setup.md", "rank": None}])
        self.assertEqual(len(q05["retrieved"]), 5)  # what was retrieved instead

    def test_text_report_lists_what_was_retrieved_instead_and_states_n(self):
        code, out, _ = run(["--queries", str(FILES / "queries.jsonl"),
                            "--results", str(FILES / "missing-document" / "results.jsonl"),
                            "--match-field", "doc_id", "--k", "1,3,5"])
        self.assertEqual(code, 1)
        self.assertIn("FAIL  never-retrieved: kb/sso-setup.md", out)
        self.assertIn('MISS    q05  "How do I set up SAML single sign-on?"', out)
        self.assertIn("relevant   kb/sso-setup.md   not in the ranking (depth 10)", out)
        self.assertIn("top 5      kb/password-reset.md#3", out)
        self.assertIn("rank 9, below k=5", out)
        rows = [line for line in out.splitlines() if line.startswith(("   1 ", "   3 ", "   5 "))]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row.endswith("   24") for row in rows))
        self.assertIn("n=24 queries", out)

    def test_duplicated_corpus_shows_as_a_regression_against_the_healthy_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "before.json"
            code, before = run_fixture("healthy")
            baseline.write_text(json.dumps(before))
            code, report = run_fixture("duplicated-corpus", "--baseline", str(baseline))
        self.assertEqual(code, 0)  # no threshold was set, so nothing blocks
        deltas = report["comparison"]["deltas"]["5"]
        self.assertLess(deltas["recall"]["change"], 0)
        self.assertLess(deltas["mrr"]["change"], 0)
        self.assertLess(deltas["precision"]["change"], 0)
        self.assertEqual(report["comparison"]["paired"]["mrr"]["worse"], 24)
        self.assertLess(report["comparison"]["paired"]["mrr"]["sign_test_p"], 0.05)
        self.assertEqual(report["comparison"]["paired"]["recall"]["sign_test_p"], 1.0)
        self.assertIn({"id": "q20", "before": "ok", "after": "miss", "worse": True},
                      [dict((k, v) for k, v in c.items() if k != "where")
                       for c in report["comparison"]["status_changes"]])

    def test_mid_sentence_snapshot_passes_retrieval_because_labels_are_document_level(self):
        code, report = run_fixture("mid-sentence")
        self.assertEqual(code, 0)
        self.assertEqual(kinds(report, "fail"), [])

    def test_threshold_blocks_only_when_not_met(self):
        code, report = run_fixture("healthy", "--min", "recall@5=0.99")
        self.assertEqual(code, 1)
        self.assertEqual(kinds(report, "fail"), ["threshold"])
        self.assertIn("n=24", report["findings"][0]["message"])
        self.assertTrue(report["findings"][0]["evidence"][0].startswith("q24 scores 0.500"))
        code, report = run_fixture("healthy", "--min", "recall@5=0.9", "--min", "mrr@5=0.9")
        self.assertEqual(code, 0)

    def test_matching_on_the_wrong_field_is_called_out(self):
        code, out, _ = run(["--queries", str(FILES / "queries.jsonl"),
                            "--results", str(FILES / "healthy" / "results.jsonl"), "--json"])
        self.assertEqual(code, 1)
        self.assertEqual(kinds(json.loads(out), "fail"), ["no-overlap"])

    def test_baseline_from_a_different_query_set_or_cutoff_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "before.json"
            baseline.write_text(json.dumps(run_fixture("healthy")[1]))
            fewer = Path(tmp) / "fewer.jsonl"
            lines = (FILES / "queries.jsonl").read_text().splitlines()
            fewer.write_text("\n".join(lines[:-1]) + "\n")
            common = ["--results", str(FILES / "healthy" / "results.jsonl"),
                      "--match-field", "doc_id", "--baseline", str(baseline)]
            code, _, err = run(["--queries", str(fewer), "--k", "1,3,5"] + common)
            self.assertEqual(code, 2)
            self.assertIn("different query set", err)
            code, _, err = run(["--queries", str(FILES / "queries.jsonl"), "--k", "10"] + common)
            self.assertEqual(code, 2)
            self.assertIn("Compare like with like", err)
            baseline.write_text("[1, 2]")
            code, _, err = run(["--queries", str(FILES / "queries.jsonl"), "--k", "1,3,5"] + common)
            self.assertEqual(code, 2)
            self.assertIn("not a retrieval_eval.py --json report", err)

    def test_output_is_deterministic(self):
        first = run_fixture("duplicated-corpus")
        second = run_fixture("duplicated-corpus")
        self.assertEqual(first, second)


class ValidateRepoTests(unittest.TestCase):
    def test_repository_passes_its_own_validator(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import validate
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = validate.main(["validate.py", str(ROOT)])
        self.assertEqual(code, 0, buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
