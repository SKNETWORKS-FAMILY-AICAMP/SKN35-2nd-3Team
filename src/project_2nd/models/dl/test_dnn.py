"""DNN 구조와 데이터 누수 방지 계약을 확인하는 경량 단위 테스트."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from deep_mlp import DeepMLP
from prepare_dnn_dataset import (
    build_feature_plan,
    encode_categorical_train_only,
    impute_nonfinite_train_median,
    snapshot_to_month_index,
    validate_split_and_target,
)
from run_shap_dnn import (
    contribution_rows,
    normalize_shap_values,
    stratified_sample_indices,
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
            "same_industry_count_300m",
            "previously_transitioned",
            "keyword_growth_score",
            "industry_historical_rate",
            "transitioned_next",
            "fold",
            "is_closed_next",
        ]

    def test_hard_leakage_and_original_category_are_always_excluded(self) -> None:
        plan, excluded = build_feature_plan(self.columns)
        features = {output for output, _, _ in plan}
        self.assertNotIn("store_id", features)
        self.assertNotIn("fold", features)
        self.assertNotIn("is_closed_next", features)
        self.assertNotIn("transitioned_next", features)
        self.assertNotIn("industry_name", features)
        self.assertIn("industry_name_enc", features)
        self.assertIn("transitioned_next", excluded)

    def test_pjw_official_features_are_retained(self) -> None:
        plan, _ = build_feature_plan(self.columns)
        features = {output for output, _, _ in plan}
        for column in (
            "previously_transitioned",
            "keyword_growth_score",
            "industry_historical_rate",
        ):
            self.assertIn(column, features)

    def test_category_mapping_is_fit_on_train_only(self) -> None:
        values = pd.Series(["가", "나", "가", "검증전용"], dtype="string")
        train_mask = np.array([True, True, True, False])
        encoded, mapping, unknown_count = encode_categorical_train_only(values, train_mask)
        self.assertEqual(set(mapping), {"가", "나"})
        self.assertEqual(float(encoded[-1]), -1.0)
        self.assertEqual(unknown_count, 1)

    def test_nonfinite_values_use_train_median(self) -> None:
        features = np.array([[1.0], [3.0], [np.nan], [np.inf]], dtype=np.float32)
        train_mask = np.array([True, True, False, False])
        result, medians, counts = impute_nonfinite_train_median(features, train_mask)
        np.testing.assert_array_equal(result[:, 0], np.array([1.0, 3.0, 2.0, 2.0]))
        self.assertEqual(medians, [2.0])
        self.assertEqual(counts, [2])

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


class ShapHelperTests(unittest.TestCase):
    def test_stratified_shap_sample_preserves_classes(self) -> None:
        labels = np.array([0] * 90 + [1] * 10, dtype=np.uint8)
        chosen = stratified_sample_indices(np.arange(100), labels, 20, seed=42)
        self.assertEqual(len(chosen), 20)
        self.assertEqual(len(np.unique(chosen)), 20)
        self.assertEqual(int(labels[chosen].sum()), 2)

    def test_shap_values_are_normalized_to_rows_by_features(self) -> None:
        values = np.zeros((5, 3, 1), dtype=np.float64)
        normalized = normalize_shap_values(values, sample_count=5, feature_count=3)
        self.assertEqual(normalized.shape, (5, 3))
        self.assertEqual(normalized.dtype, np.float32)

    def test_shap_contributions_are_ranked_by_direction(self) -> None:
        rows = contribution_rows(
            ["a", "b", "c"],
            np.array([1.0, 2.0, 3.0]),
            np.array([-0.3, 0.1, 0.5]),
            "up",
            limit=2,
        )
        self.assertEqual([row["feature"] for row in rows], ["c", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
