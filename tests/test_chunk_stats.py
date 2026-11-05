import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "rag-pipeline-review"
FILES = SKILL / "evals" / "files"
sys.path.insert(0, str(SKILL / "scripts"))

import chunk_stats  # noqa: E402

SENTENCE = "This paragraph explains one thing clearly and then it stops at the end."


def run(argv):
    """Run the script's main(); return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = chunk_stats.main(argv)
        except SystemExit as exc:  # argparse usage errors
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def run_fixture(name, *extra):
    code, out, _ = run(["--chunks", str(FILES / name / "chunks.jsonl"),
                        "--docs", str(FILES / "docs.json"), "--json"] + list(extra))
    return code, json.loads(out)


def by_kind(report):
    return dict((f["kind"], f) for f in report["findings"])


def chunk(cid, doc, text):
    return {"id": cid, "doc_id": doc, "text": text}


def unique(n):
    """A complete sentence that shares no 3-word shingle with unique(m) for m != n."""
    return "Topic %d covers alpha%d beta%d gamma%d delta%d epsilon%d zeta%d in full." % ((n,) * 7)


class HelperTests(unittest.TestCase):
    def test_normalise_collapses_whitespace_and_case(self):
        self.assertEqual(chunk_stats.normalise("  Hello\n\n  WORLD \t"), "hello world")

    def test_percentile_uses_nearest_rank(self):
        ten = list(range(1, 11))
        self.assertEqual(chunk_stats.percentile(ten, 5), 1)    # ceil(0.5) = 1st value
        self.assertEqual(chunk_stats.percentile(ten, 95), 10)  # ceil(9.5) = 10th value
        twenty = list(range(1, 21))
        self.assertEqual(chunk_stats.percentile(twenty, 95), 19)
        self.assertEqual(chunk_stats.percentile(twenty, 5), 1)
        self.assertEqual(chunk_stats.percentile([7], 95), 7)

    def test_distribution(self):
        d = chunk_stats.distribution([4, 1, 3, 2])
        self.assertEqual((d["min"], d["median"], d["mean"], d["max"]), (1, 2.5, 2.5, 4))

    def test_shingles(self):
        self.assertEqual(chunk_stats.shingles("a b c d"), {"a b c", "b c d"})
        self.assertEqual(chunk_stats.shingles("a b"), {"a b"})
        self.assertEqual(chunk_stats.shingles("..."), set())

    def test_near_duplicate_jaccard_is_exact(self):
        # Shingles: {123, 234, 345, 456} and {123, 234, 345, 457}: 3 shared of 5 distinct = 0.6
        a, b = "one two three four five six", "one two three four five seven"
        pairs = chunk_stats.near_duplicates([a, b, "something else entirely here now"], 0.6)
        self.assertEqual([(i, j) for i, j, _ in pairs], [(0, 1)])
        self.assertAlmostEqual(pairs[0][2], 0.6)
        self.assertEqual(chunk_stats.near_duplicates([a, b], 0.61), [])

    def test_near_duplicate_search_matches_brute_force(self):
        rng = random.Random(7)
        vocabulary = ["w%d" % i for i in range(40)]
        texts = []
        for _ in range(40):
            words = [rng.choice(vocabulary) for _ in range(rng.randint(3, 30))]
            texts.append(" ".join(words))
            for _ in range(rng.randint(0, 2)):  # add edited copies
                copy = list(words)
                for _ in range(rng.randint(1, 4)):
                    copy[rng.randrange(len(copy))] = rng.choice(vocabulary)
                texts.append(" ".join(copy))
        sets = [chunk_stats.shingles(t) for t in texts]
        found_any = False
        for threshold in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0):
            expected = set()
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    union = len(sets[i] | sets[j])
                    if union and len(sets[i] & sets[j]) / float(union) >= threshold - 1e-12:
                        expected.add((i, j))
            got = set((i, j) for i, j, _ in chunk_stats.near_duplicates(texts, threshold))
            self.assertEqual(got, expected, "threshold %s" % threshold)
            found_any = found_any or bool(expected)
        self.assertTrue(found_any)

    def test_ends_open(self):
        open_endings = [
            "This sentence was cut somewhere in the",
            "The refund goes back to the original payment method (see the billing page",
            "A heading\n\nand then a paragraph that stops without any",
        ]
        closed_endings = [
            "This sentence is complete.",
            "Is this a question?",
            'She said "stop."',
            "(See the billing page.)",
            "The steps are:",
            "## A heading with no full stop",
            "Intro.\n- a list item with no full stop at all here",
            "Intro.\n3. a numbered step with no full stop here",
            "| plan | price |",
            "Example:\n    return the value without punctuation",
            "```\ncode\n```",
            "",
            "   \n",
        ]
        for text in open_endings:
            self.assertTrue(chunk_stats.ends_open(text), text)
        for text in closed_endings:
            self.assertFalse(chunk_stats.ends_open(text), text)

    def test_starts_lower(self):
        self.assertTrue(chunk_stats.starts_lower("and then it continues"))
        self.assertTrue(chunk_stats.starts_lower('  "quoted continuation'))
        self.assertFalse(chunk_stats.starts_lower("A new sentence"))
        self.assertFalse(chunk_stats.starts_lower("42 is a number"))
        self.assertFalse(chunk_stats.starts_lower(""))


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name, rows):
        path = self.dir / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return str(path)

    def report(self, rows, *extra):
        code, out, err = run(["--chunks", self.write("chunks.jsonl", rows), "--json"] + list(extra))
        return code, (json.loads(out) if out else None), err

    def test_clean_chunks_produce_no_finding(self):
        code, report, _ = self.report([chunk("a#%d" % n, "a", unique(n)) for n in range(6)])
        self.assertEqual(code, 0)
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["stats"]["chunks"], 6)

    def test_empty_chunk_is_an_error_with_its_line(self):
        code, report, _ = self.report([chunk("a#0", "a", unique(0)), chunk("a#1", "a", "  \n ")])
        self.assertEqual(code, 1)
        finding = by_kind(report)["empty"]
        self.assertEqual(finding["level"], "error")
        self.assertEqual(finding["items"][0]["id"], "a#1")
        self.assertTrue(finding["items"][0]["where"].endswith("chunks.jsonl:2"))

    def test_duplicate_chunk_id_is_an_error(self):
        code, report, _ = self.report([chunk("a#0", "a", unique(0)), chunk("a#0", "a", unique(1))])
        self.assertEqual(code, 1)
        self.assertIn("chunks.jsonl:2", by_kind(report)["duplicate-id"]["items"][0]["detail"])

    def test_exact_duplicate_ignores_case_and_whitespace(self):
        code, report, _ = self.report([
            chunk("a#0", "a", SENTENCE), chunk("a#1", "a", unique(1)),
            chunk("b#0", "b", "  " + SENTENCE.upper().replace(" ", "\n")), chunk("b#1", "b", unique(2))])
        self.assertEqual(code, 1)
        finding = by_kind(report)["exact-duplicate"]
        self.assertEqual(finding["count"], 1)
        self.assertEqual(finding["items"][0]["id"], "a#0")
        self.assertIn("b#0", finding["items"][0]["detail"])
        self.assertIn("1 redundant chunk(s), 25.0% of all", finding["summary"])
        self.assertNotIn("duplicate-document", by_kind(report))

    def test_text_shared_by_three_different_documents_is_boilerplate(self):
        footer = "Copyright Example Ltd. All rights reserved. Do not distribute this page."
        rows = []
        for n, doc in enumerate(("a", "b", "c")):
            rows += [chunk("%s#0" % doc, doc, unique(n)), chunk("%s#1" % doc, doc, footer)]
        code, report, _ = self.report(rows)
        self.assertEqual(code, 1)
        found = by_kind(report)
        self.assertEqual(found["boilerplate"]["count"], 1)
        self.assertNotIn("exact-duplicate", found)
        self.assertNotIn("duplicate-document", found)

    def test_the_same_document_ingested_three_times_is_not_called_boilerplate(self):
        rows = []
        for doc in ("a", "copy-1/a", "copy-2/a"):
            rows += [chunk("%s#0" % doc, doc, unique(0)), chunk("%s#1" % doc, doc, unique(1))]
        code, report, _ = self.report(rows)
        self.assertEqual(code, 1)
        found = by_kind(report)
        self.assertNotIn("boilerplate", found)
        self.assertEqual(found["exact-duplicate"]["count"], 2)
        self.assertEqual(found["duplicate-document"]["count"], 1)
        self.assertEqual(found["duplicate-document"]["items"][0]["doc"], "a")
        self.assertIn("copy-1/a", found["duplicate-document"]["items"][0]["detail"])
        self.assertIn("copy-2/a", found["duplicate-document"]["items"][0]["detail"])

    def test_tiny_and_oversized_are_warnings(self):
        rows = [chunk("a#%d" % n, "a", unique(n)) for n in range(5)]
        rows.append(chunk("a#5", "a", "Too short."))
        rows.append(chunk("a#6", "a", " ".join(unique(n) for n in range(100, 110))))
        code, report, _ = self.report(rows)
        self.assertEqual(code, 0)
        found = by_kind(report)
        self.assertEqual([i["id"] for i in found["tiny"]["items"]], ["a#5"])
        self.assertEqual([i["id"] for i in found["oversized"]["items"]], ["a#6"])
        self.assertEqual(found["tiny"]["level"], "warn")
        code, _, _ = self.report(rows, "--fail-on", "warn")
        self.assertEqual(code, 1)

    def test_max_chars_sets_an_absolute_limit(self):
        rows = [chunk("a#%d" % n, "a", unique(n)) for n in range(5)]
        code, report, _ = self.report(rows, "--max-chars", "40")
        self.assertEqual(by_kind(report)["oversized"]["count"], 5)
        self.assertIn("--max-chars 40", by_kind(report)["oversized"]["items"][0]["detail"])

    def test_near_duplicate_is_a_warning_with_both_chunks_named(self):
        base = "Refunds for annual plans can be requested within 30 days of the charge by the owner."
        rows = [chunk("a#0", "a", base), chunk("b#0", "b", base.replace("owner", "admin")),
                chunk("c#0", "c", unique(3))]
        code, report, _ = self.report(rows, "--near-dup-threshold", "0.8")
        self.assertEqual(code, 0)
        item = by_kind(report)["near-duplicate"]["items"][0]
        self.assertEqual(item["id"], "a#0")
        self.assertIn("b#0", item["detail"])
        # 14 shingles each, 13 shared (only the last differs): 13 / 15 = 0.87
        self.assertIn("0.87 similar", item["detail"])

    def test_near_duplicate_check_can_be_skipped_by_size(self):
        code, report, _ = self.report([chunk("a#%d" % n, "a", unique(n)) for n in range(4)],
                                      "--near-dup-limit", "3")
        self.assertEqual(code, 0)
        self.assertIn("skipped", report["stats"]["near_duplicate_check"])

    def test_mid_sentence_is_a_warning_when_rare_and_an_error_when_common(self):
        rows = [chunk("a#%d" % n, "a", unique(n)) for n in range(9)]
        rows.append(chunk("a#9", "a", "This last chunk was cut off by the splitter right in the"))
        code, report, _ = self.report(rows)
        self.assertEqual(code, 0)
        finding = by_kind(report)["mid-sentence"]
        self.assertEqual((finding["level"], finding["count"]), ("warn", 1))
        self.assertEqual(report["stats"]["mid_sentence_share"], 0.1)
        code, report, _ = self.report(rows, "--max-cut-share", "0.1")
        self.assertEqual(code, 1)
        self.assertEqual(by_kind(report)["mid-sentence"]["level"], "error")
        code, report, _ = self.report(rows, "--no-sentence-check")
        self.assertNotIn("mid-sentence", by_kind(report))

    def test_short_tail_counts_only_when_the_next_chunk_continues_in_lower_case(self):
        head = unique(1) + "\n\nOnly an"
        cut = [chunk("a#0", "a", head), chunk("a#1", "a", "owner can change the role. " + unique(2))]
        code, report, _ = self.report(cut)
        item = by_kind(report)["mid-sentence"]["items"][0]
        self.assertEqual(item["id"], "a#0")
        self.assertIn("next chunk starts", item["detail"])
        heading = [chunk("a#0", "a", unique(1) + "\n\nRefund policy"), chunk("a#1", "a", unique(2))]
        code, report, _ = self.report(heading)
        self.assertNotIn("mid-sentence", by_kind(report))
        other_doc = [chunk("a#0", "a", head), chunk("b#0", "b", "owner can change the role. " + unique(2))]
        code, report, _ = self.report(other_doc)
        self.assertNotIn("mid-sentence", by_kind(report))

    def test_docs_list_finds_documents_with_no_chunks_and_unlisted_documents(self):
        docs = self.dir / "docs.json"
        docs.write_text(json.dumps(["a", {"id": "b"}]))
        rows = [chunk("a#0", "a", unique(0)), chunk("z#0", "z", unique(1))]
        code, report, _ = self.report(rows, "--docs", str(docs))
        self.assertEqual(code, 1)
        found = by_kind(report)
        self.assertEqual(found["no-chunks"]["items"][0]["doc"], "b")
        self.assertTrue(found["no-chunks"]["items"][0]["where"].endswith("docs.json[1]"))
        self.assertEqual(found["unlisted-document"]["items"][0]["doc"], "z")
        self.assertEqual(found["unlisted-document"]["level"], "warn")

    def test_custom_and_nested_field_names(self):
        rows = [{"chunk_id": n, "page_content": unique(n), "metadata": {"source": "a.md"}} for n in range(3)]
        code, report, _ = self.report(rows, "--id-field", "chunk_id", "--text-field", "page_content",
                                      "--doc-field", "metadata.source")
        self.assertEqual(code, 0)
        self.assertEqual(report["stats"]["documents"], 1)

    def test_bad_inputs_exit_two_with_a_message_and_no_traceback(self):
        cases = [
            ([{"id": "a", "doc_id": "a"}], "--text-field"),
            ([{"id": "a", "text": "x"}], "--doc-field"),
            ([{"doc_id": "a", "text": "x"}], "--id-field"),
            ([{"id": "a", "doc_id": "a", "text": 5}], "must be a string"),
            (["just a string"], "must be a JSON object"),
            ([], "no records"),
        ]
        for rows, expected in cases:
            code, report, err = self.report(rows)
            self.assertEqual(code, 2, expected)
            self.assertIsNone(report)
            self.assertIn(expected, err)
            self.assertNotIn("Traceback", err)

    def test_broken_file_missing_file_and_bad_options_exit_two(self):
        path = self.dir / "broken.jsonl"
        path.write_text('{"id": "a", "doc_id": "a", "text": "x"}\n{oops\n')
        code, _, err = run(["--chunks", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("broken.jsonl:2", err)
        self.assertEqual(run(["--chunks", str(self.dir / "none.jsonl")])[0], 2)
        self.assertEqual(run(["--chunks", str(path), "--near-dup-threshold", "0"])[0], 2)
        self.assertEqual(run([])[0], 2)

    def test_byte_order_mark_and_non_ascii_text_on_an_ascii_terminal(self):
        path = self.dir / "chunks.jsonl"
        row = chunk("a#0", "a", "Gr\u00fc\u00dfe")
        path.write_bytes(b"\xef\xbb\xbf" + json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")
        raw = io.BytesIO()
        terminal = io.TextIOWrapper(raw, encoding="ascii")
        with redirect_stdout(terminal):
            code = chunk_stats.main(["--chunks", str(path)])
            terminal.flush()
        self.assertEqual(code, 0)
        self.assertIn(b"tiny (1)", raw.getvalue())
        self.assertIn(b"Gr\\xfc\\xdfe", raw.getvalue())

    def test_yaml_without_pyyaml_exits_two_and_says_how_to_convert(self):
        path = self.dir / "chunks.yaml"
        path.write_text("- id: a\n  doc_id: a\n  text: hello\n")
        saved = sys.modules.get("yaml", "absent")
        sys.modules["yaml"] = None  # makes "import yaml" raise ImportError
        try:
            code, _, err = run(["--chunks", str(path)])
        finally:
            if saved == "absent":
                del sys.modules["yaml"]
            else:
                sys.modules["yaml"] = saved
        self.assertEqual(code, 2)
        self.assertIn("PyYAML is not installed", err)
        self.assertIn("JSON", err)


class FixtureTests(unittest.TestCase):
    def test_healthy_snapshot_has_no_finding(self):
        code, report = run_fixture("healthy")
        self.assertEqual(code, 0)
        self.assertEqual(report["findings"], [])
        self.assertEqual((report["stats"]["chunks"], report["stats"]["documents"]), (48, 12))

    def test_duplicated_corpus_is_reported(self):
        code, report = run_fixture("duplicated-corpus")
        self.assertEqual(code, 1)
        found = by_kind(report)
        # 48 chunks were ingested twice; one archived paragraph differs by one number.
        self.assertEqual(found["exact-duplicate"]["count"], 47)
        self.assertIn("47 redundant chunk(s), 49.0% of all", found["exact-duplicate"]["summary"])
        self.assertEqual(found["duplicate-document"]["count"], 11)
        copies = " ".join(i["doc"] + " " + i["detail"] for i in found["duplicate-document"]["items"])
        self.assertIn("kb-archive/api-keys.md", copies)
        self.assertNotIn("refunds.md", copies)
        self.assertEqual(found["near-duplicate"]["count"], 1)
        pair = found["near-duplicate"]["items"][0]
        self.assertEqual(pair["id"], "kb/refunds.md#0")
        self.assertIn("kb-archive/refunds.md#0", pair["detail"])
        self.assertEqual(found["unlisted-document"]["count"], 12)
        self.assertNotIn("boilerplate", found)
        for finding in report["findings"]:
            for item in finding["items"]:
                self.assertIn("chunks.jsonl:", item["where"])

    def test_mid_sentence_chunking_is_reported(self):
        code, report = run_fixture("mid-sentence")
        self.assertEqual(code, 1)
        finding = by_kind(report)["mid-sentence"]
        self.assertEqual((finding["level"], finding["count"]), ("error", 22))
        self.assertEqual(report["stats"]["mid_sentence_share"], round(22 / 36.0, 4))
        self.assertEqual(finding["items"][0]["id"], "kb/refunds.md#0")
        self.assertIn("next chunk starts 'funds go back", finding["items"][0]["detail"])
        flagged = set(i["id"] for i in finding["items"])
        # These two chunks happen to end exactly on a full stop, and last chunks end the document.
        self.assertNotIn("kb/password-reset.md#0", flagged)
        self.assertNotIn("kb/account-deletion.md#1", flagged)
        self.assertNotIn("kb/refunds.md#2", flagged)
        self.assertEqual(sorted(by_kind(report)), ["mid-sentence", "tiny"])

    def test_document_with_no_chunks_is_reported(self):
        code, report = run_fixture("missing-document")
        self.assertEqual(code, 1)
        self.assertEqual(sorted(by_kind(report)), ["no-chunks"])
        item = by_kind(report)["no-chunks"]["items"][0]
        self.assertEqual(item["doc"], "kb/sso-setup.md")
        self.assertIn("docs.json[2]", item["where"])

    def test_text_report_names_evidence(self):
        code, out, _ = run(["--chunks", str(FILES / "mid-sentence" / "chunks.jsonl")])
        self.assertEqual(code, 1)
        self.assertIn("ERROR mid-sentence (22): chunks that end without sentence punctuation, "
                      "61.1% of 36 non-empty chunks", out)
        self.assertIn("chunks.jsonl:1]", out)
        self.assertIn("RESULT: 1 error finding(s), 1 warning finding(s)", out)

    def test_output_is_deterministic(self):
        self.assertEqual(run_fixture("duplicated-corpus"), run_fixture("duplicated-corpus"))


if __name__ == "__main__":
    unittest.main()
