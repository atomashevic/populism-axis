"""Tests for the anchor-revision pilot: candidate anchors, NLI dedup/remap, adoption rule, data scope."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_anchor_balance  # noqa: E402
import pilot_anchor_revision as pilot  # noqa: E402

V4_PATH = ROOT / "anchors" / "proposition_anchors.json"
V5_GREEN_PATH = ROOT / "anchors" / "candidates" / "v5_green.json"
V5_RED_PATH = ROOT / "anchors" / "candidates" / "v5_red.json"
V5_GREEN_RED_PATH = ROOT / "anchors" / "candidates" / "v5_green_red.json"
RUNTIME_MANIFEST_PATH = ROOT / "results" / "frozen_reference" / "runtime_manifest.json"

class CandidateAnchorTests(unittest.TestCase):
    def test_frozen_anchors_unchanged(self) -> None:
        manifest = json.loads(RUNTIME_MANIFEST_PATH.read_text(encoding="utf-8"))
        entries = {entry["path"]: entry for entry in manifest["assets"]}
        for rel_path in ("anchors/proposition_anchors.json", "anchors/claim_hypotheses.json"):
            entry = entries[rel_path]
            raw = (ROOT / rel_path).read_bytes()
            self.assertEqual(len(raw), entry["bytes"], rel_path)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"], rel_path)

    def test_candidates_form_a_two_by_two_design(self) -> None:
        """Wordings are settled in the review document, not here; the test fixes the design.

        v5_green revises populist sentences and may revise contrast sentences; v5_red revises
        contrast sentences only; v5_green_red carries both sets of revisions and nothing else.
        """
        v4 = json.loads(V4_PATH.read_text(encoding="utf-8"))
        arms = {
            "v5_green": json.loads(V5_GREEN_PATH.read_text(encoding="utf-8")),
            "v5_red": json.loads(V5_RED_PATH.read_text(encoding="utf-8")),
            "v5_green_red": json.loads(V5_GREEN_RED_PATH.read_text(encoding="utf-8")),
        }
        for name, payload in arms.items():
            self.assertEqual(payload["schema"], "populism_axis.v3.proposition_anchors.v1", name)
            self.assertEqual(payload["anchor_set_id"], f"pa_{name}_pilot_r1", name)
            self.assertIs(payload["outcome_labels_used"], False, name)
            self.assertEqual(payload["construct_source"], v4["construct_source"], name)
            self.assertTrue(str(payload.get("rationale", "")).strip(), name)
            self.assertEqual(set(payload["channels"]), {"P", "E", "M"}, name)
            self.assertEqual(payload["channels"]["M"], v4["channels"]["M"], f"{name}.M must match frozen V4")
            for ch in ("P", "E"):
                block = payload["channels"][ch]
                self.assertEqual(block["label"], v4["channels"][ch]["label"], f"{name}.{ch}")
                for pole in ("populist", "mainstream"):
                    self.assertEqual(len(block[pole]), 10, f"{name}.{ch}.{pole}")
                    self.assertTrue(all(isinstance(s, str) and s.strip() for s in block[pole]))
                    self.assertEqual(len(set(block[pole])), 10, f"{name}.{ch}.{pole} repeats a sentence")

        def revised(name: str) -> dict[tuple[str, str, int], str]:
            return {
                (ch, pole, idx): text
                for ch in ("P", "E")
                for pole in ("populist", "mainstream")
                for idx, text in enumerate(arms[name]["channels"][ch][pole])
                if text != v4["channels"][ch][pole][idx]
            }

        green, red, both = revised("v5_green"), revised("v5_red"), revised("v5_green_red")
        self.assertTrue(green and red)
        self.assertTrue(any(pole == "populist" for _, pole, _ in green))
        self.assertTrue(all(pole == "mainstream" for _, pole, _ in red), "v5_red must keep V4's populist sentences")
        self.assertFalse(set(green) & set(red), "a slot is revised by both factors")
        self.assertEqual(both, {**green, **red}, "v5_green_red must be exactly v5_green plus v5_red")

    def test_e_channel_balance_gates_and_no_near_duplicates(self) -> None:
        # v5_red keeps V4's populist E sentences and with them V4's failed they/them gate,
        # so the E gates are asserted for the two arms that revise the populist pole.
        for path in (V5_GREEN_PATH, V5_RED_PATH, V5_GREEN_RED_PATH):
            report = check_anchor_balance.check(path)
            if path != V5_RED_PATH:
                e_group = report["groups"]["E"]
                self.assertTrue(e_group["gated"], str(path))
                self.assertTrue(all(e_group["gates"].values()), f"{path}: failed E gates {e_group['gates']}")
            for ch in ("P", "E", "M"):
                self.assertEqual(report["groups"][ch]["near_duplicates_left"], [], f"{path}.{ch}")
                mainstream = json.loads(path.read_text(encoding="utf-8"))["channels"][ch]["mainstream"]
                ids_and_texts = [(f"{ch}-{i:02d}", text) for i, text in enumerate(mainstream)]
                self.assertEqual(check_anchor_balance.near_duplicates(ids_and_texts), [], f"{path}.{ch}.mainstream")


class DeduplicationAndRemapTests(unittest.TestCase):
    def test_roundtrip_partial_overlap_premises_and_hypotheses(self) -> None:
        # Two mock arms with partially overlapping premises (shared, arm-unique, and repeated)
        # AND partially different top-5 P and top-5 E hypotheses, using non-contiguous doc_indices.
        rows = [
            # Arm 1: doc 10 (2 windows), doc 20 (1 window); doc 15 missing to test zero-fill
            {"arm": "arm1", "dataset": "ds_a", "doc_index": 10, "doc_id": "d10", "window_rank": 0, "premise": "Shared window alpha."},
            {"arm": "arm1", "dataset": "ds_a", "doc_index": 10, "doc_id": "d10", "window_rank": 1, "premise": "Arm1 only window beta."},
            {"arm": "arm1", "dataset": "ds_a", "doc_index": 20, "doc_id": "d20", "window_rank": 0, "premise": "Shared cross-doc window gamma."},
            # Arm 2: shares alpha and gamma (even at different doc/rank), plus Arm2-only delta and repeated alpha
            {"arm": "arm2", "dataset": "ds_a", "doc_index": 10, "doc_id": "d10", "window_rank": 0, "premise": "Shared cross-doc window gamma."},
            {"arm": "arm2", "dataset": "ds_a", "doc_index": 10, "doc_id": "d10", "window_rank": 1, "premise": "Arm2 only window delta."},
            {"arm": "arm2", "dataset": "ds_a", "doc_index": 20, "doc_id": "d20", "window_rank": 0, "premise": "Shared window alpha."},
            # Second dataset sharing text "Shared window alpha." across datasets too
            {"arm": "arm2", "dataset": "ds_b", "doc_index": 5, "doc_id": "b5", "window_rank": 0, "premise": "Shared window alpha."},
        ]
        window_rows = pd.DataFrame(rows)

        # Arm 1 & Arm 2: 3 shared P, 2 distinct P -> 7 union P; 2 shared E, 3 distinct E -> 8 union E
        arm1_hyps = [
            {"hypothesis_id": "P00", "channel": "P", "text": "P shared 1"},
            {"hypothesis_id": "P01", "channel": "P", "text": "P shared 2"},
            {"hypothesis_id": "P02", "channel": "P", "text": "P shared 3"},
            {"hypothesis_id": "P03", "channel": "P", "text": "P arm1 only A"},
            {"hypothesis_id": "P04", "channel": "P", "text": "P arm1 only B"},
            {"hypothesis_id": "E00", "channel": "E", "text": "E shared 1"},
            {"hypothesis_id": "E01", "channel": "E", "text": "E shared 2"},
            {"hypothesis_id": "E02", "channel": "E", "text": "E arm1 only A"},
            {"hypothesis_id": "E03", "channel": "E", "text": "E arm1 only B"},
            {"hypothesis_id": "E04", "channel": "E", "text": "E arm1 only C"},
        ]
        # Notice arm2 also assigns different arm-local hypothesis_id to a shared text ("P09" -> "P shared 1")
        arm2_hyps = [
            {"hypothesis_id": "P09", "channel": "P", "text": "P shared 1"},
            {"hypothesis_id": "P01", "channel": "P", "text": "P shared 2"},
            {"hypothesis_id": "P05", "channel": "P", "text": "P arm2 only A"},
            {"hypothesis_id": "P02", "channel": "P", "text": "P shared 3"},
            {"hypothesis_id": "P07", "channel": "P", "text": "P arm2 only B"},
            {"hypothesis_id": "E08", "channel": "E", "text": "E arm2 only A"},
            {"hypothesis_id": "E00", "channel": "E", "text": "E shared 1"},
            {"hypothesis_id": "E09", "channel": "E", "text": "E arm2 only B"},
            {"hypothesis_id": "E01", "channel": "E", "text": "E shared 2"},
            {"hypothesis_id": "E06", "channel": "E", "text": "E arm2 only C"},
        ]
        arm_hyps = {"arm1": arm1_hyps, "arm2": arm2_hyps}

        requests_df, arm_window_map, union_hyps, mapped_arm_hyps = pilot.deduplicate_nli_requests(
            window_rows, arm_hyps
        )

        self.assertEqual(list(requests_df.columns), ["request_id", "premise"])
        self.assertNotIn("premise", arm_window_map.columns)
        self.assertEqual(len(requests_df), 4)
        self.assertEqual(len(arm_window_map), len(window_rows))

        # Check union hypotheses counts (7 P + 8 E = 15) and unique IDs
        p_union = [h for h in union_hyps if h["channel"] == "P"]
        e_union = [h for h in union_hyps if h["channel"] == "E"]
        self.assertEqual(len(p_union), 7)
        self.assertEqual(len(e_union), 8)
        self.assertEqual([h["hypothesis_id"] for h in p_union], [f"U_P{i:02d}" for i in range(7)])
        self.assertEqual([h["hypothesis_id"] for h in e_union], [f"U_E{i:02d}" for i in range(8)])

        # Verify request_id mapping round-trips every original premise exactly
        req_to_premise = dict(zip(requests_df["request_id"].astype(int), requests_df["premise"].astype(str)))
        reconstructed = [req_to_premise[int(rid)] for rid in arm_window_map["request_id"]]
        self.assertEqual(reconstructed, [r["premise"] for r in rows])

        # Verify mapped_arm_hypotheses point to the exact union text
        for arm_name, hyps in mapped_arm_hyps.items():
            for h in hyps:
                u_item = union_hyps[h["union_index"]]
                self.assertEqual(u_item["hypothesis_id"], h["union_hypothesis_id"])
                self.assertEqual(u_item["channel"], h["channel"])
                self.assertEqual(u_item["text"], h["text"], arm_name)

        # Build deterministic entailment table keyed by (premise, channel, hyp_text)
        rng = np.random.default_rng(12345)
        truth_probs = {}
        for _, req_row in requests_df.iterrows():
            prem = str(req_row["premise"])
            for h in union_hyps:
                truth_probs[(prem, h["channel"], h["text"])] = float(rng.uniform(0.01, 0.99))

        entail_probs = np.zeros((len(requests_df), len(union_hyps)), dtype=np.float64)
        for r_idx, req_row in requests_df.iterrows():
            prem = str(req_row["premise"])
            for h_idx, h in enumerate(union_hyps):
                entail_probs[r_idx, h_idx] = truth_probs[(prem, h["channel"], h["text"])]

        union_ids = [h["hypothesis_id"] for h in union_hyps]
        req_ids = requests_df["request_id"].to_numpy(dtype=np.int64)

        # Remap for each arm & compare against brute-force direct calculation
        for arm_name, orig_hyps in arm_hyps.items():
            sub_map = arm_window_map[(arm_window_map["arm"] == arm_name) & (arm_window_map["dataset"] == "ds_a")]
            doc_indices = np.array([10, 15, 20], dtype=np.int64)
            for k in (1, 2):
                doc_scores, win_df = pilot.remap_arm_nli_scores(
                    sub_map,
                    req_ids,
                    entail_probs,
                    union_ids,
                    mapped_arm_hyps[arm_name],
                    doc_indices,
                    k=k,
                )

                # Brute-force per-window expectation from raw rows
                orig_arm_rows = [r for r in rows if r["arm"] == arm_name and r["dataset"] == "ds_a"]
                self.assertEqual(len(win_df), len(orig_arm_rows))
                expected_win = []
                for orig_r, w_out in zip(orig_arm_rows, win_df.itertuples()):
                    p_vals = [
                        (h["hypothesis_id"], truth_probs[(orig_r["premise"], "P", h["text"])])
                        for h in orig_hyps if h["channel"] == "P"
                    ]
                    e_vals = [
                        (h["hypothesis_id"], truth_probs[(orig_r["premise"], "E", h["text"])])
                        for h in orig_hyps if h["channel"] == "E"
                    ]
                    best_p_id, max_p = max(p_vals, key=lambda x: x[1])
                    best_e_id, max_e = max(e_vals, key=lambda x: x[1])
                    pe = float(np.sqrt(max_p * max_e))
                    self.assertAlmostEqual(w_out.P_nli, max_p, places=12)
                    self.assertAlmostEqual(w_out.E_nli, max_e, places=12)
                    self.assertAlmostEqual(w_out.PE_nli, pe, places=12)
                    self.assertEqual(w_out.P_best, best_p_id)
                    self.assertEqual(w_out.E_best, best_e_id)
                    expected_win.append((orig_r["doc_index"], orig_r["window_rank"], max_p, max_e, pe))

                # Brute-force document pooling
                for d_pos, doc_idx in enumerate(doc_indices):
                    kept = [w for w in expected_win if w[0] == doc_idx and w[1] < k]
                    if not kept:
                        self.assertEqual(doc_scores["P_nli"][d_pos], 0.0)
                        self.assertEqual(doc_scores["E_nli"][d_pos], 0.0)
                        self.assertEqual(doc_scores["PE_nli"][d_pos], 0.0)
                    else:
                        self.assertAlmostEqual(doc_scores["P_nli"][d_pos], np.mean([w[2] for w in kept]), places=12)
                        self.assertAlmostEqual(doc_scores["E_nli"][d_pos], np.mean([w[3] for w in kept]), places=12)
                        self.assertAlmostEqual(doc_scores["PE_nli"][d_pos], np.mean([w[4] for w in kept]), places=12)


class AdoptionRuleTests(unittest.TestCase):
    @staticmethod
    def _delta(value: float, low: float, high: float) -> dict:
        return {"delta_spearman": value, "ci95": [low, high]}

    def test_rule_of_the_development_protocol(self) -> None:
        d = self._delta
        passing = pilot.adoption_reading(d(0.010, -0.05, 0.07), d(-0.010, -0.03, 0.01))
        self.assertTrue(passing["passes"])
        self.assertAlmostEqual(passing["mean_delta"], 0.0)
        self.assertFalse(pilot.adoption_reading(d(0.009, -0.05, 0.07), d(0.02, 0.0, 0.04))["passes"])
        self.assertFalse(pilot.adoption_reading(d(0.03, -0.01, 0.07), d(-0.011, -0.03, 0.01))["passes"])
        # An interval entirely below zero fails even when the point differences pass.
        self.assertFalse(pilot.adoption_reading(d(0.03, -0.01, 0.07), d(-0.005, -0.009, -0.001))["passes"])
        self.assertFalse(pilot.adoption_reading(d(0.012, -0.04, -0.001), d(0.01, -0.01, 0.03))["passes"])


class DataScopeTests(unittest.TestCase):
    def test_only_development_sets_are_named(self) -> None:
        self.assertEqual(pilot.DATASETS, ("english_dev", "agora"))
        source = (ROOT / "scripts" / "pilot_anchor_revision.py").read_text(encoding="utf-8")
        for fragment in (
            "english_holdout",
            "embeddings_eu",
            "embeddings_external",
            "requests_eu",
            "scores_eu",
            "results/final",
            '"final"',
            "usgov",
            "popbert",
            "portugal",
            "antielite",
            "reproduce_table",
        ):
            self.assertNotIn(fragment, source, f"the pilot must not read held-out or external data: {fragment}")


if __name__ == "__main__":
    unittest.main()
