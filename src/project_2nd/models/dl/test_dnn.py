"""DNN 구조와 데이터 누수 방지 계약을 확인하는 경량 단위 테스트."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from deep_mlp import DeepMLP
from prepare_dnn_dataset import (
    build_feature_plan,
    snapshot_to_month_index,
    validate_split_and_target,
)
from train_dnn import classification_metrics, choose_f1_threshold


class DeepMLPTests(unittest.TestCase):
    def test_output_shape_is_one_logit_per_row(self) -> None:
        model = DeepMLP(input_dim=34)
        model.eval()
        result = model(torch.zeros(8, 34))
        self.assertEqual(tuple(result.shape), (8,))

    def test_parameter_count_matches_handoff_for_34_inputs(self) -> None:
        model = DeepMLP(input_dim=34)
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(count, 51_073)


class DatasetContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = [
            "snapshot_date",
            "store_id",
            "industry_name",
            "industry_name_enc",
            "same_industry_count_300m",
            "previously_transitioned",
            "keyword_growth_score",
            "industry_historical_rate",
            "transitioned_next",
            "fold",
            "is_closed_next",
        ]

    def test_hard_leakage_and_original_category_are_always_excluded(self) -> None:
        for policy in ("official", "time_safe"):
            plan, excluded = build_feature_plan(self.columns, policy)
            features = {output for output, _ in plan}
            self.assertNotIn("store_id", features)
            self.assertNotIn("fold", features)
            self.assertNotIn("is_closed_next", features)
            self.assertNotIn("transitioned_next", features)
            self.assertNotIn("industry_name", features)
            self.assertIn("industry_name_enc", features)
            self.assertIn("transitioned_next", excluded)

    def test_time_safe_policy_excludes_suspected_time_leakage(self) -> None:
        plan, excluded = build_feature_plan(self.columns, "time_safe")
        features = {output for output, _ in plan}
        for column in (
            "previously_transitioned",
            "keyword_growth_score",
            "industry_historical_rate",
        ):
            self.assertNotIn(column, features)
            self.assertEqual(excluded[column], "potential-time-leakage")

    def test_snapshot_month_spacing_is_preserved(self) -> None:
        result = snapshot_to_month_index(["202312", "202406", "202412", "202506"])
        np.testing.assert_array_equal(result, np.array([0, 6, 12, 18], dtype=np.float32))

    def test_split_and_binary_target_contract(self) -> None:
        split = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
        target = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        validate_split_and_target(split, target)


class MetricTests(unittest.TestCase):
    def test_threshold_is_selected_from_scores(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.uint8)
        scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        threshold, best_f1 = choose_f1_threshold(labels, scores)
        self.assertGreaterEqual(threshold, 0.2)
        self.assertAlmostEqual(best_f1, 1.0)
        metrics = classification_metrics(labels, scores, threshold)
        self.assertAlmostEqual(float(metrics["f1"]), 1.0)
        self.assertTrue(np.isfinite(float(metrics["pr_auc"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
