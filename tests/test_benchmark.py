"""Checks for the experiment's data and evaluation boundaries."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark import SEED, ROOT, load_data, preprocessing


class BenchmarkTests(unittest.TestCase):
    def test_data_contract_and_ablation(self):
        scenarios, audit = load_data()
        self.assertEqual(audit.raw_rows.tolist(), [2111, 569])
        self.assertEqual(scenarios["breast_cancer"][0].shape[1], 30)
        full, y, _ = scenarios["obesity_full"]
        reduced, reduced_y, _ = scenarios["obesity_without_body_size"]
        self.assertEqual(set(full.columns) - set(reduced.columns), {"Height", "Weight"})
        np.testing.assert_array_equal(y, reduced_y)
        self.assertNotIn("id", scenarios["breast_cancer"][0].columns)

    def test_splits_are_disjoint_and_shared_for_ablation(self):
        scenarios, _ = load_data()
        saved = []
        for name, (X, y, labels) in scenarios.items():
            train, test = train_test_split(np.arange(len(y)), stratify=y, test_size=.2, random_state=SEED)
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(train) + len(test), len(X))
            self.assertEqual(set(y[test]), set(range(len(labels))))
            if name.startswith("obesity"):
                saved.append(test)
        np.testing.assert_array_equal(*saved)

    def test_preprocessing_handles_unseen_category_without_learning_it(self):
        train = pd.DataFrame({"age": [20., 40., np.nan], "transport": ["walk", "bus", "walk"]})
        held_out = pd.DataFrame({"age": [100.], "transport": ["train"]})
        transform = preprocessing(train).fit(train)
        result = transform.transform(held_out)
        self.assertTrue(np.isfinite(result).all())
        categories = transform.named_transformers_["categorical"].named_steps["encode"].categories_[0]
        self.assertNotIn("train", categories)
        self.assertEqual(transform.named_transformers_["numeric"].named_steps["impute"].statistics_[0], 30.)


if __name__ == "__main__":
    unittest.main()
