# Findings from the revised benchmark

The run uses a fixed seed of 42, a stratified 80/20 development/test split and five development folds. The following are the selected models; all candidate fold scores are retained in `cv_folds.csv`.

| Scenario | Selected model | Development macro-F1 | Test accuracy | Test macro-F1 |
| --- | --- | --- | --- | --- |
| Obesity · all inputs | LightGBM | 96.87% ± 0.95% | 96.65% | 96.64% |
| Obesity · without height/weight | CatBoost | 83.20% ± 3.70% | 85.89% | 85.56% |
| Breast cancer | Logistic regression | 97.16% ± 1.81% | 96.49% | 96.19% |

## Test uncertainty

| Scenario | Test rows | Accuracy | 95% bootstrap interval | Macro-F1 | 95% bootstrap interval |
| --- | --- | --- | --- | --- | --- |
| Obesity · all inputs | 418 | 96.65% | 94.98%–98.33% | 96.64% | 94.98%–98.31% |
| Obesity · without height/weight | 418 | 85.89% | 82.54%–89.23% | 85.56% | 82.28%–88.90% |
| Breast cancer | 114 | 96.49% | 92.98%–99.12% | 96.19% | 92.21%–99.06% |

Intervals are percentile intervals from 1,000 within-class bootstrap resamples of the held-out predictions. They preserve the observed class mix and condition on the fitted model and fixed split. They omit variation from refitting, selecting a model, collecting new data and changing the synthetic composition. They should not be used to declare significance between the candidates.

## Body measurements change the obesity task

The strongest full-input development candidate is LightGBM. When height and weight are removed, all three boosting candidates lose substantial macro-F1 on the same folds:

| Model | Full-input CV macro-F1 | Without body-size CV macro-F1 | Drop (percentage points) |
| --- | --- | --- | --- |
| XGBoost | 95.83% | 82.92% | 12.91 |
| LightGBM | 96.87% | 82.80% | 14.06 |
| CatBoost | 95.99% | 83.20% | 12.78 |

This is evidence that performance depends on the available input measurements. It is not a causal result, a future-risk prediction study, or a way to validate the dataset's labels independently. Remaining attributes such as age and family history are still present in the reduced scenario.

The selected full-input test model makes 14 mistakes in 418 rows. The selected reduced-input model makes 59. In the reduced-input confusion matrix, Overweight II has the weakest recall: 40 of 58 rows are classified correctly (68.97%). The same-split ablation is useful, but the selected models differ; the per-model development table above is the controlled comparison.

## A simple cancer baseline is competitive

Logistic regression wins by the unrounded development mean, although its score rounds to the same value as CatBoost. There is no repeated-seed or paired statistical study here to support a claim that either is consistently better.

The held-out cancer matrix is:

| True class | Predicted benign | Predicted malignant |
| --- | --- | --- |
| Benign | 71 | 1 |
| Malignant | 3 | 39 |

Malignant recall is 39/42 = 92.86%. ROC AUC is 0.996, but a strong ranking score does not remove the three missed malignant examples at the default decision threshold. No threshold was selected against this test set.

The 114-row test sample is small. One additional mistake changes accuracy by about 0.88 percentage points. Interpretation should keep that scale in mind.

## Which inputs the selected models rely on

Permutation importance is measured on original columns using held-out macro-F1, with 15 shuffles per feature. The three highest mean drops are:

- **Obesity · all inputs:** Weight (0.765), Height (0.284), Gender (0.116).
- **Obesity · without height/weight:** Age (0.219), Gender (0.214), FCVC (0.139).
- **Breast cancer:** texture_worst (0.029), concavity_worst (0.024), symmetry_worst (0.019).

These are predictive reliance measures, not causal effects. Correlated measurements can substitute for one another, while negative or tiny importances can reflect noise. Test-set interpretation is performed after selection and is not used to revise the model or feature set.

## Runtime and depth

Training time includes preprocessing and model fitting, excludes inference, and uses four threads for each boosting model. `training_time.png` shows measured mean times per fold, not numbers copied from the class notebook.

The LightGBM depth comparison uses depths 3, 6 and 9 on the development data. It describes sensitivity under the fixed settings and does not replace the reported model-selection rule. Since `num_leaves` remains at its default, increasing `max_depth` may not increase realized tree complexity.

## What would strengthen the study

Repeated development splits would show whether the close model rankings are stable. Nested cross-validation would be appropriate for a larger tuning search. A genuinely new external cohort would provide stronger validation than another split of the supplied benchmarks. For obesity, known synthetic lineage or a separately collected real-only dataset would help address dependence across splits.

The synthetic-data caveat is documented by [UCI](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). Dataset provenance and licenses are in `data/README.md`.
