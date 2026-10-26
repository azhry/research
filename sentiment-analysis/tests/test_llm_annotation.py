"""Deterministic unit tests for the DeepSeek pre-annotation pipeline.

These tests never call a model: every model interaction is faked through the
same validation/export code paths used by real runs.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from aste_repro import llm_annotation as la


def make_args(tmp: Path, **overrides: object) -> argparse.Namespace:
    args = argparse.Namespace(
        dataset="all",
        split="all",
        sources_dir=tmp / "sources",
        labelstudio_dir=tmp / "labelstudio",
        output_dir=tmp / "out",
        cache_dir=tmp / "cache",
        provider="opencode",
        model="deepseek-v4.1-flash-free",
        variant="max",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class PromptTests(unittest.TestCase):
    def test_prompt_is_stable_and_contains_every_sentence(self) -> None:
        texts = ["kamar nya bagus .", "ac nya rusak ."]
        prompt = la.build_prompt(texts)
        self.assertIn("1. kamar nya bagus .", prompt)
        self.assertIn("2. ac nya rusak .", prompt)
        self.assertNotIn("__SENTENCES__", prompt)
        self.assertEqual(prompt, la.build_prompt(texts))
        self.assertEqual(len(la.prompt_sha256()), 64)
        self.assertIn("prompt-v1", la.model_version_label("opencode", "deepseek-v4.1-flash-free", "max"))

    def test_run_key_is_safe_and_namespaced(self) -> None:
        key = la.run_key("opencode", "deepseek-v4.1-flash-free", "max")
        self.assertRegex(key, r"^[A-Za-z0-9._-]+$")
        self.assertTrue(key.startswith("opencode_deepseek-v4.1-flash-free-max-"))
        self.assertNotEqual(key, la.run_key("opencode-go", "deepseek-v4.1-flash", "max"))


class JsonExtractionTests(unittest.TestCase):
    def test_fenced_and_prose_wrapped_arrays(self) -> None:
        raw = 'Here you go:\n```json\n[{"id": 1, "triplets": []}]\n```\nDone.'
        self.assertEqual(la.extract_triplet_array(raw), [{"id": 1, "triplets": []}])
        raw = 'Result: [{"id": 1, "triplets": [{"aspect": null, "opinion": "bagus", "polarity": "POS"}]}]'
        self.assertEqual(len(la.extract_triplet_array(raw)), 1)

    def test_invalid_payloads_raise(self) -> None:
        with self.assertRaises(ValueError):
            la.extract_triplet_array("no array here")
        with self.assertRaises(ValueError):
            la.extract_triplet_array('[{"id": 1,}]')
        with self.assertRaises(ValueError):
            la.extract_triplet_array('{"id": 1}')


class ResolveSpanTests(unittest.TestCase):
    def test_prefers_token_aligned_occurrence(self) -> None:
        text = "kenakalan enak"
        span, occurrences, issue = la.resolve_span(text, "enak")
        self.assertEqual(span, (10, 14))
        self.assertEqual(occurrences, 2)
        self.assertIsNone(issue)

    def test_falls_back_to_first_occurrence(self) -> None:
        span, occurrences, issue = la.resolve_span("Pelayannya", "yan")
        self.assertEqual(span, (4, 7))
        self.assertEqual(occurrences, 1)
        self.assertIsNone(issue)

    def test_unresolved_and_trimmed_spans(self) -> None:
        span, occurrences, issue = la.resolve_span("kamar bagus", "pelayanan")
        self.assertIsNone(span)
        self.assertEqual(occurrences, 0)
        self.assertIn("exact substring", issue or "")
        span, occurrences, issue = la.resolve_span("kamar bagus", " kamar ")
        self.assertEqual(span, (0, 5))
        self.assertIn("trimmed", issue or "")


class ResolveBatchTests(unittest.TestCase):
    def test_resolves_spans_and_normalizes_polarity(self) -> None:
        sentences = [la.Sentence("casa", "test", 0, "Mobil ini irit dan nyaman .")]
        payload = [
            {
                "id": 1,
                "triplets": [
                    {"aspect": "Mobil ini", "opinion": "irit", "polarity": "positive"},
                    {"aspect": "Mobil ini", "opinion": "nyaman", "polarity": "POS"},
                    {"aspect": None, "opinion": "irit", "polarity": "POS"},
                ],
            }
        ]
        records, errors = la.resolve_batch(sentences, payload)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        triplets = records[0]["triplets"]
        self.assertEqual(len(triplets), 3)
        self.assertEqual(triplets[0]["polarity"], "POS")
        self.assertEqual(triplets[0]["aspect_span"], [0, 9])
        self.assertEqual(triplets[0]["opinion_span"], [10, 14])
        self.assertIsNone(triplets[2]["aspect_span"])
        self.assertEqual(la.batch_problems(records, errors), [])

    def test_missing_and_duplicate_ids_are_errors(self) -> None:
        sentences = [
            la.Sentence("hoasa", "test", 0, "kamar nya luas ."),
            la.Sentence("hoasa", "test", 1, "ac nya rusak ."),
        ]
        records, errors = la.resolve_batch(sentences, [{"id": 1, "triplets": []}])
        self.assertEqual(records, [])
        self.assertTrue(any("missing ids" in error for error in errors))
        records, errors = la.resolve_batch(sentences, [{"id": 1, "triplets": []}, {"id": 1, "triplets": []}])
        self.assertTrue(any("duplicate id" in error for error in errors))

    def test_bad_triplets_are_recorded_not_guessed(self) -> None:
        sentences = [la.Sentence("casa", "test", 0, "Mobil ini irit .")]
        payload = [
            {
                "id": 1,
                "triplets": [
                    {"aspect": "Mobil ini", "opinion": "irit", "polarity": "MAYBE"},
                    {"aspect": None, "opinion": None, "polarity": "POS"},
                    {"aspect": "Mobil ini", "opinion": "kencang", "polarity": "POS"},
                ],
            }
        ]
        records, errors = la.resolve_batch(sentences, payload)
        self.assertEqual(errors, [])
        issues = records[0]["issues"]
        self.assertTrue(any("unknown polarity" in issue for issue in issues))
        self.assertTrue(any("neither aspect nor opinion" in issue for issue in issues))
        triplets = records[0]["triplets"]
        self.assertEqual(len(triplets), 1)
        self.assertIn("opinion: unresolved", triplets[0]["issues"])
        problems = la.batch_problems(records, errors)
        self.assertTrue(any("unresolved span" in problem for problem in problems))

    def test_duplicate_triplets_are_deduplicated(self) -> None:
        sentences = [la.Sentence("casa", "test", 0, "Mobil ini irit .")]
        payload = [
            {
                "id": 1,
                "triplets": [
                    {"aspect": "Mobil ini", "opinion": "irit", "polarity": "POS"},
                    {"aspect": "Mobil ini", "opinion": "irit", "polarity": "POS"},
                ],
            }
        ]
        records, _ = la.resolve_batch(sentences, payload)
        self.assertEqual(len(records[0]["triplets"]), 1)


class PredictionResultTests(unittest.TestCase):
    def test_label_studio_result_shape(self) -> None:
        text = "Mobil ini irit ."
        triplets = [
            {"aspect_span": [0, 9], "opinion_span": [10, 14], "polarity": "POS"},
            {"aspect_span": None, "opinion_span": [10, 14], "polarity": "POS"},
        ]
        result = la.prediction_result(triplets, text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["id"], "aspect_0")
        self.assertEqual(result[0]["value"]["text"], "Mobil ini")
        self.assertEqual(result[1]["value"]["text"], "irit")
        self.assertEqual(result[1]["value"]["labels"], ["Opinion"])
        self.assertEqual(result[2]["from_id"], "aspect_0")
        self.assertEqual(result[2]["to_id"], "opinion_0")
        self.assertEqual(result[2]["type"], "relation")
        self.assertEqual(result[2]["direction"], "right")
        self.assertEqual(result[2]["labels"], ["POS"])

    def test_empty_triplets_yield_empty_result(self) -> None:
        self.assertEqual(la.prediction_result([], "kamar nya luas ."), [])


class PairResolutionTests(unittest.TestCase):
    def test_dataset_split_pairs(self) -> None:
        self.assertEqual(
            la.resolve_pairs("hoasa", "all"),
            [("hoasa", "train"), ("hoasa", "dev"), ("hoasa", "test")],
        )
        self.assertEqual(la.resolve_pairs("casa", "valid"), [("casa", "valid")])
        self.assertEqual(
            la.resolve_pairs("all", "train"), [("hoasa", "train"), ("casa", "train")]
        )
        with self.assertRaises(SystemExit):
            la.resolve_pairs("hoasa", "valid")


class GoldSpanTests(unittest.TestCase):
    def test_gold_char_spans_include_implicit_aspects(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "hoasa").mkdir(parents=True)
            la.write_text(
                root / "hoasa" / "test.txt",
                "kamar nya luas .####[([0, 1], [2], 'POS'), ([-1], [0], 'NEG')]\n",
            )
            gold = la.hoasa_gold_spans(root, "test")
        self.assertEqual(
            gold[0],
            [([0, 9], [10, 14], "POS"), (None, [0, 5], "NEG")],
        )


class ExportVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.args = make_args(root)
        (self.args.sources_dir / "hoasa").mkdir(parents=True)
        (self.args.sources_dir / "casa").mkdir(parents=True)
        self.args.labelstudio_dir.mkdir(parents=True)

        la.write_text(
            self.args.sources_dir / "hoasa" / "test.txt",
            "\n".join(
                [
                    "kamar nya luas .####[([0, 1], [2], 'POS')]",
                    "ac nya rusak .####[([-1], [2], 'NEG')]",
                    "pelayanan ramah .####[([0], [1], 'POS')]",
                ]
            )
            + "\n",
        )

        rows = ["Mobil ini irit dan nyaman .", "Harga nya murah .", "Pelayanan nya buruk ."]
        csv_lines = ["sentence,fuel,machine,others,part,price,service"]
        for row in rows:
            csv_lines.append(row + "," + ",".join(["positive"] * 6))
        la.write_text(
            self.args.sources_dir / "casa" / "test_preprocess.csv", "\n".join(csv_lines) + "\n"
        )

        self.original_casa = [
            {"data": {"text": rows[0]}, "annotations": []},
            {
                "data": {"text": rows[1]},
                "annotations": [
                    {
                        "result": [
                            {
                                "from_name": "label",
                                "id": "aspect_0",
                                "to_name": "text",
                                "type": "labels",
                                "value": {
                                    "start": 0,
                                    "end": 5,
                                    "text": "Harga",
                                    "labels": ["Aspect"],
                                },
                            },
                            {
                                "from_name": "label",
                                "id": "opinion_0",
                                "to_name": "text",
                                "type": "labels",
                                "value": {
                                    "start": 10,
                                    "end": 15,
                                    "text": "murah",
                                    "labels": ["Opinion"],
                                },
                            },
                            {
                                "from_id": "aspect_0",
                                "to_id": "opinion_0",
                                "type": "relation",
                                "direction": "right",
                                "labels": ["POS"],
                            },
                        ]
                    }
                ],
            },
            {"data": {"text": rows[2]}, "annotations": []},
        ]
        la.write_text(
            self.args.labelstudio_dir / "casa_test.json",
            json.dumps(self.original_casa, ensure_ascii=False, indent=1) + "\n",
        )

        run_dir = Path(self.args.cache_dir) / la.run_key(self.args.provider, self.args.model, self.args.variant)
        self._write_batch(
            run_dir,
            "hoasa",
            "test",
            [
                {"id": 1, "triplets": [{"aspect": "kamar nya", "opinion": "luas", "polarity": "POS"}]},
                {"id": 2, "triplets": [{"aspect": None, "opinion": "rusak", "polarity": "NEG"}]},
                {"id": 3, "triplets": [{"aspect": "pelayanan", "opinion": "ramah", "polarity": "POS"}]},
            ],
        )
        self._write_batch(
            run_dir,
            "casa",
            "test",
            [
                {
                    "id": 1,
                    "triplets": [
                        {"aspect": "Mobil ini", "opinion": "irit", "polarity": "POS"},
                        {"aspect": None, "opinion": "nyaman", "polarity": "POS"},
                    ],
                },
                {"id": 2, "triplets": [{"aspect": "Harga nya", "opinion": "murah", "polarity": "POS"}]},
                {"id": 3, "triplets": []},
            ],
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_batch(self, run_dir: Path, dataset: str, split: str, payload: list[dict]) -> None:
        sentences = la.load_sentences(dataset, split, self.args.sources_dir)
        records, errors = la.resolve_batch(sentences, payload)
        self.assertEqual(errors, [])
        for record, sentence in zip(records, sentences):
            self.assertEqual(record["index"], sentence.index)
        batch = {
            "status": "ok",
            "batch_index": 0,
            "sentence_indices": [sentence.index for sentence in sentences],
            "records": records,
        }
        la.write_json(run_dir / f"{dataset}_{split}" / "batch_00000.json", batch)

    def test_export_verify_roundtrip_preserves_human_annotations(self) -> None:
        la.export_all([("hoasa", "test")], self.args)
        la.export_all([("casa", "test")], self.args)
        code = la.verify_directory(self.args.output_dir, self.args.labelstudio_dir)
        self.assertEqual(code, 0)
        merged = la.read_json(self.args.output_dir / "casa_test_merged.json")
        self.assertEqual(merged[1]["annotations"], self.original_casa[1]["annotations"])
        self.assertEqual(merged[1]["data"], self.original_casa[1]["data"])
        self.assertEqual(len(merged[1]["predictions"]), 1)
        sidecar = (self.args.output_dir / "hoasa_test_triplets.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(sidecar), 3)
        first = json.loads(sidecar[0])
        self.assertEqual(first["triplets"][0]["aspect_span"], [0, 9])
        implicit = json.loads(sidecar[1])
        self.assertIsNone(implicit["triplets"][0]["aspect_span"])

    def test_verify_detects_span_corruption(self) -> None:
        la.export_all([("hoasa", "test")], self.args)
        path = self.args.output_dir / "hoasa_test.json"
        tasks = la.read_json(path)
        tasks[0]["predictions"][0]["result"][0]["value"]["text"] = "KAMAR"
        la.write_text(path, json.dumps(tasks, ensure_ascii=False, indent=1) + "\n")
        errors: list[str] = []
        la.verify_file(path, self.args.labelstudio_dir, errors)
        self.assertTrue(any("span text mismatch" in error for error in errors))

    def test_missing_batches_are_detected(self) -> None:
        args = make_args(Path(self._tmp.name) / "gap-root")
        args.sources_dir = self.args.sources_dir
        args.labelstudio_dir = self.args.labelstudio_dir
        la.export_all([("casa", "test")], args)
        code = la.verify_directory(args.output_dir, args.labelstudio_dir)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
