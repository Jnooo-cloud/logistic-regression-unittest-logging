from __future__ import annotations

import unittest
import warnings

import numpy as np

from advertising_ml_system import (
    benchmark_fit_runtime,
    evaluate_predictions,
    load_full_dataset,
    load_test_dataset,
    split_dataset,
    train_model,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


class TestAdvertisingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full_df = load_full_dataset()
        X_train, _, y_train, _ = split_dataset(full_df)
        cls.X_train = X_train
        cls.y_train = y_train
        cls.model = train_model(X_train, y_train)
        cls.test_df = load_test_dataset()

    def test_predict_function_on_saved_test_data(self):
        evaluation = evaluate_predictions(self.model, self.test_df)

        self.assertGreaterEqual(evaluation["accuracy"], 0.95)
        self.assertEqual(len(evaluation["confusion_matrix"]), 2)
        self.assertEqual(
            sum(sum(row) for row in evaluation["confusion_matrix"]), len(self.test_df)
        )

    def test_fit_runtime_is_within_120_percent_of_reference(self):
        repetitions = 500

        benchmark_fit_runtime(self.X_train, self.y_train, repetitions)
        baseline = benchmark_fit_runtime.last_run_seconds
        threshold = baseline * 1.2

        benchmark_fit_runtime(self.X_train, self.y_train, repetitions)
        current_runtime = benchmark_fit_runtime.last_run_seconds

        self.assertLessEqual(
            current_runtime,
            threshold,
            msg=(
                f"Current fit runtime {current_runtime:.6f}s exceeds the allowed "
                f"threshold of {threshold:.6f}s (120% of {baseline:.6f}s)."
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
