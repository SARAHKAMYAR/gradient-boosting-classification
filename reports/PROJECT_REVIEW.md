# From the class notebook to the portfolio version

## What the original project did

The original notebook compared XGBoost, LightGBM and CatBoost on an obesity dataset and the Wisconsin breast cancer dataset. It used five-fold stratified cross-validation, reported weighted classification metrics, plotted confusion matrices, and explored LightGBM tree depth.

That is a useful starting point. The portfolio revision keeps the comparison and adds a more explicit experimental design and a readable explanation of the results.

## What changed and why

| Original issue | Revision |
| --- | --- |
| A comparison cell referred to `mushroom_results_df`, which was not defined in this project. Another cell used a presentation table before creating it. | A single script owns the experiment; the walkthrough reads its saved outputs in order. |
| The notebook contained only code cells, repeated plots and repeated table formatting. | A focused walkthrough explains the question, data, evaluation, results and limitations. |
| Accuracy and runtime charts used manually typed numbers, including inconsistent runtime values. | All figures are drawn from the run's metrics or predictions. |
| Cross-validation used the entire dataset before a later train/test split. | The revised pipeline reserves 20% before comparing models on the remaining 80%. |
| LightGBM was called the best model directly. | The winner is chosen by mean development macro-F1, without looking at the revised test metrics. |
| Weighted recall duplicated accuracy for these single-label classification problems. | The comparison emphasizes macro-F1 and retains accuracy and weighted-F1; the cancer report also records malignant recall and ROC AUC. |
| There were no simple comparison models. | Majority-class and scaled logistic-regression baselines provide context. |
| Categorical encoding was done before splitting. | Imputation, scaling and one-hot encoding are fit inside each training fold. Pre-split one-hot encoding did not use target values, but it still learned the full category vocabulary. |
| The same fitted model instance was reused across folds. | Each fold receives a fresh cloned pipeline. |
| Fold variability was not shown. | Mean and sample standard deviation are saved and plotted. Fold SD is not a confidence interval or evidence of statistical significance. |
| Obesity body measurements were not examined separately. | A matched-split ablation removes height and weight and repeats the comparison. This is a different input scenario, not proof of causation. |
| Feature importance used LightGBM split counts. | Permutation importance measures the change in macro-F1 when a raw input column is shuffled, with variability across 15 repeats. |
| Global warning suppression hid diagnostics. | The revised code leaves warnings visible. |
| There was no dataset attribution or reusable setup. | Data notes, pinned packages, saved split indices, input hashes and run metadata make the benchmark easier to inspect and rerun. |

## Verification performed

- Ran the full five-fold experiment with 500 boosting iterations for every candidate in all three scenarios.
- Checked the supplied cancer feature values and labels against scikit-learn's canonical dataset; they match.
- Checked the obesity CSV against UCI's canonical download; it matches.
- Removed exact duplicate obesity rows before splitting and saved the count in `data_audit.csv`.
- Ran the data contract, split separation, shared ablation split, unseen-category and training-only imputation checks.
- Executed the walkthrough from beginning to end and checked it for error outputs.
- Inspected the generated figure layouts and corrected confusion-matrix tick label rotation.
- Checked result tables, prediction counts, probability sums, model selection and README figure paths.

The GitHub Actions workflow is included but has not run on GitHub yet. It will run after the repository is uploaded.

## What this revision does not establish

The original notebook had already evaluated the supplied datasets. The revised test split is excluded from selection in this new pipeline; it is not a newly collected or historically unseen cohort. Results should be treated as a retrospective benchmark.

Exact deduplication cannot identify synthetic relatives in the obesity data. Similar rows can remain across splits. A second dataset with known grouping or a new independent cohort would be needed for stronger evidence of generalization.

This is a reproducible portfolio benchmark. It does not demonstrate clinical readiness or production deployment. The code uses the same one-hot preprocessing for each candidate; it does not test CatBoost's native categorical support or claim to optimize every library equally.

## Sensible next experiments

1. Repeat the development comparison with several split seeds to check how stable the close rankings are.
2. Use nested cross-validation if adding a substantial hyperparameter search.
3. Compare CatBoost's native categorical handling with the shared one-hot setup.
4. For cancer, use a larger independent cohort before considering calibration, threshold selection or any clinical interpretation. Thresholds should be selected on development data, then assessed once on the independent test data.

These are proposed next steps, not completed experiments.
