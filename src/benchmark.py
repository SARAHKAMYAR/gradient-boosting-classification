"""Reproducible tabular classification benchmark. Run from the repository root."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             classification_report, recall_score, roc_auc_score, average_precision_score,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SEED = 42
COLORS = ["#8C98A5", "#D79B32", "#2878A0", "#249B85", "#805BB3"]
NAMES = ["Majority baseline", "Logistic regression", "XGBoost", "LightGBM", "CatBoost"]
OBESITY_LABELS = ["Insufficient", "Normal", "Overweight I", "Overweight II",
                  "Obesity I", "Obesity II", "Obesity III"]
OBESITY_CLASSES = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
                   "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]


def load_data():
    obesity_raw = pd.read_csv(ROOT / "data/obesity.csv")
    cancer_raw = pd.read_csv(ROOT / "data/breast_cancer.csv")
    audit = []
    cleaned = []
    for name, raw, target, labels in [
        ("obesity", obesity_raw, "NObeyesdad", OBESITY_CLASSES),
        ("breast_cancer", cancer_raw, "diagnosis", ["B", "M"]),
    ]:
        data = raw.drop(columns=["id"], errors="ignore").dropna(axis=1, how="all")
        duplicates = int(data.duplicated().sum())
        data = data.drop_duplicates().reset_index(drop=True)
        if not set(data[target]).issubset(labels):
            raise ValueError(f"Unexpected labels in {name}")
        X = data.drop(columns=target)
        y = data[target].map({label: i for i, label in enumerate(labels)}).to_numpy()
        audit.append(dict(dataset=name, raw_rows=len(raw), rows=len(data),
                          exact_duplicates_removed=duplicates, features=X.shape[1],
                          missing_values=int(X.isna().sum().sum())))
        cleaned.append((X, y))
    Xo, yo = cleaned[0]
    Xc, yc = cleaned[1]
    scenarios = {
        "obesity_full": (Xo, yo, OBESITY_LABELS),
        "obesity_without_body_size": (Xo.drop(columns=["Height", "Weight"]), yo, OBESITY_LABELS),
        "breast_cancer": (Xc, yc, ["Benign", "Malignant"]),
    }
    return scenarios, pd.DataFrame(audit)


def preprocessing(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], sparse_threshold=0)


def models(X, iterations=500):
    estimators = [
        DummyClassifier(strategy="most_frequent"),
        LogisticRegression(max_iter=3000, random_state=SEED),
        XGBClassifier(n_estimators=iterations, max_depth=6, learning_rate=.05,
                      tree_method="hist", random_state=SEED, n_jobs=4),
        LGBMClassifier(n_estimators=iterations, max_depth=6, learning_rate=.05,
                       random_state=SEED, n_jobs=4, verbosity=-1),
        CatBoostClassifier(iterations=iterations, depth=6, learning_rate=.05,
                           random_seed=SEED, thread_count=4, verbose=False,
                           allow_writing_files=False),
    ]
    return {name: Pipeline([("preprocess", preprocessing(X)), ("model", estimator)])
            for name, estimator in zip(NAMES, estimators)}


def scores(y, prediction, probability):
    result = dict(accuracy=accuracy_score(y, prediction),
                  macro_f1=f1_score(y, prediction, average="macro", zero_division=0),
                  weighted_f1=f1_score(y, prediction, average="weighted", zero_division=0))
    if probability.shape[1] == 2:
        result.update(malignant_recall=recall_score(y, prediction, pos_label=1, zero_division=0),
                      roc_auc=roc_auc_score(y, probability[:, 1]))
    return result


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion(scenario, winner, ytest, pred, labels):
    cm = confusion_matrix(ytest, pred, labels=np.arange(len(labels)))
    normal = cm / cm.sum(axis=1, keepdims=True)
    annotation = np.array([[f"{cm[i,j]}\n{normal[i,j]:.0%}" for j in range(len(labels))] for i in range(len(labels))])
    fig, ax = plt.subplots(figsize=(9, 7.5) if len(labels) > 2 else (6, 5), layout="constrained")
    sns.heatmap(normal, annot=annotation, fmt="", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"label": "Share of true class"})
    ax.set(title=f"{scenario.replace('_', ' ').title()} · {winner}\n20% held-out split · revised experiment",
           xlabel="Predicted class", ylabel="True class")
    ax.tick_params(axis="x", rotation=40 if len(labels) > 2 else 0)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, f"{scenario}_confusion")


def plot_cancer_curves(ytest, probability, winner):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    RocCurveDisplay.from_predictions(ytest, probability, name=winner, ax=axes[0],
                                    curve_kwargs={"color": COLORS[3], "label": f"{winner} (AUC = {roc_auc_score(ytest, probability):.3f})"})
    axes[0].plot([0, 1], [0, 1], "--", color="#999999")
    PrecisionRecallDisplay.from_predictions(ytest, probability, name=winner, ax=axes[1],
                                           curve_kwargs={"color": COLORS[3], "label": f"{winner} (AP = {average_precision_score(ytest, probability):.3f})"})
    axes[1].axhline(ytest.mean(), ls="--", color="#999999", label="Malignant prevalence")
    axes[1].legend()
    axes[0].set_title("Malignant class · ROC")
    axes[1].set_title("Malignant class · precision–recall")
    save_figure(fig, "breast_cancer_curves")


def plot_comparison(summary, scenarios):
    titles = {
        "obesity_full": "Obesity · all inputs",
        "obesity_without_body_size": "Obesity · without height and weight",
        "breast_cancer": "Breast cancer",
    }
    # Stack panels so labels remain readable at notebook and README widths.
    scenarios = list(scenarios)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10.5, 3.3 * len(scenarios)), squeeze=False)
    fig.subplots_adjust(left=.22, right=.76, top=.90, bottom=.08, hspace=.65)
    fig.suptitle("Model comparison", fontsize=19, fontweight="bold", y=.98)
    fig.text(.5, .945, "Five development folds · error bars show fold standard deviation",
             ha="center", fontsize=11, color="#52606D")
    for ax, scenario in zip(axes.ravel(), scenarios):
        part = summary[summary.scenario == scenario].set_index("model").loc[NAMES]
        ax.barh(NAMES, part.macro_f1_mean, height=.62, xerr=part.macro_f1_std,
                color=COLORS, capsize=3, error_kw={"ecolor": "#273444", "elinewidth": 1.2})
        for i, (value, std) in enumerate(zip(part.macro_f1_mean, part.macro_f1_std)):
            ax.text(1.025, i, f"{value:.2%} ± {std:.2%}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=11, color="#273444", clip_on=False)
        ax.text(1.025, 1.055, "Mean ± SD", transform=ax.transAxes,
                fontsize=10, color="#52606D", clip_on=False)
        ax.set(xlim=(0, 1), title=titles.get(scenario, scenario.replace("_", " ").title()))
        ax.set_title(ax.get_title(), loc="left", fontsize=13, fontweight="bold", pad=15)
        ax.set_title("", loc="center")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis="both", labelsize=11)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E2E8EE", linewidth=.8)
        ax.grid(axis="y", visible=False)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.invert_yaxis()
    axes[-1, 0].set_xlabel("Development macro-F1", fontsize=12, labelpad=10)
    save_figure(fig, "model_comparison")


def bootstrap_interval(y, prediction):
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(1000):
        # Resample within each class; preserve the observed class composition.
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), size=(y == c).sum(), replace=True)
                              for c in np.unique(y)])
        values.append([accuracy_score(y[idx], prediction[idx]),
                       f1_score(y[idx], prediction[idx], average="macro", zero_division=0)])
    low, high = np.quantile(values, [.025, .975], axis=0)
    return dict(accuracy_ci_low=low[0], accuracy_ci_high=high[0],
                macro_f1_ci_low=low[1], macro_f1_ci_high=high[1])


def plot_importance(importance, scenario):
    """Show original-column permutation effects without hiding small or negative values."""
    friendly = {
        "family_history_with_overweight": "Family history of overweight",
        "FAVC": "High-calorie food frequency", "FCVC": "Vegetable consumption",
        "NCP": "Number of main meals", "CAEC": "Eating between meals",
        "CH2O": "Water consumption", "SCC": "Calorie monitoring",
        "FAF": "Physical activity", "TUE": "Time using technology",
        "CALC": "Alcohol consumption", "MTRANS": "Transportation method",
        "SMOKE": "Smoking",
    }
    titles = {"obesity_full": "Obesity · all inputs",
              "obesity_without_body_size": "Obesity · without height and weight",
              "breast_cancer": "Breast cancer"}
    top = importance.sort_values("mean", ascending=False).head(10).reset_index(drop=True)
    labels = [friendly.get(f, f.replace("_", " ").replace(" se", " (standard error)")
                           .replace(" worst", " (worst)").replace(" mean", " (mean)").capitalize())
              for f in top.feature]
    mean, std = top["mean"].to_numpy() * 100, top["std"].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=.32, right=.77, top=.82, bottom=.23)
    fig.suptitle(titles.get(scenario, scenario), fontsize=18, fontweight="bold", y=.97)
    fig.text(.5, .915, "Permutation importance · top 10 features", ha="center", fontsize=12, color="#52606D")
    ax.errorbar(mean, np.arange(len(top)), xerr=std, fmt="o", markersize=7,
                color=COLORS[3], ecolor="#52606D", elinewidth=1.4, capsize=3)
    ax.set_yticks(np.arange(len(top)), labels=labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#7A8793", linestyle="--", linewidth=1)
    ax.set_xlabel("Drop in test macro-F1 (percentage points)", fontsize=11, labelpad=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E2E8EE", linewidth=.8)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.text(1.04, 1.04, "Mean ± SD (pp)", transform=ax.transAxes, fontsize=10, color="#52606D")
    for i, (value, spread) in enumerate(zip(mean, std)):
        ax.text(1.04, i, f"{value:.3f} ± {spread:.3f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=10, color="#273444", clip_on=False)
    fig.text(.32, .025, "Dots: mean of 15 shuffles. Whiskers: ±1 shuffle SD, not confidence intervals.\n"
             "Values near zero or below zero can reflect noise or redundant inputs.",
             fontsize=9, color="#52606D", va="bottom")
    save_figure(fig, f"{scenario}_importance")


def run(iterations=500):
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlepad": 16,
                         "grid.alpha": .2, "svg.fonttype": "none"})
    scenarios, audit = load_data()
    audit.to_csv(REPORTS / "data_audit.csv", index=False)
    folds, summaries, tests, depth_rows, splits = [], [], [], [], {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, key in zip(axes, ["obesity_full", "breast_cancer"]):
        X, y, labels = scenarios[key]
        counts = np.bincount(y)
        ax.barh(labels, counts, color=COLORS[2])
        for i, count in enumerate(counts):
            ax.text(count + 5, i, str(count), va="center")
        ax.set_xlim(0, max(counts) * 1.2)
        ax.invert_yaxis()
        ax.set(title="Obesity levels" if key == "obesity_full" else "Breast cancer diagnosis",
               xlabel="Rows after exact duplicate removal")
    save_figure(fig, "class_distribution")

    for scenario, (X, y, labels) in scenarios.items():
        print(f"\n{scenario}: {len(X)} rows", flush=True)
        train, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=SEED)
        splits[scenario] = {"development": train.tolist(), "test": test.tolist()}
        Xdev, ydev = X.iloc[train], y[train]
        Xtest, ytest = X.iloc[test], y[test]
        cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xdev, ydev))
        candidates = models(Xdev, iterations)
        for name, pipeline in candidates.items():
            rows = []
            for fold, (tr, va) in enumerate(cv, 1):
                estimator = clone(pipeline)
                start = time.perf_counter()
                estimator.fit(Xdev.iloc[tr], ydev[tr])
                elapsed = time.perf_counter() - start
                prediction = np.asarray(estimator.predict(Xdev.iloc[va])).ravel().astype(int)
                row = dict(scenario=scenario, model=name, fold=fold, fit_seconds=elapsed,
                           **scores(ydev[va], prediction, estimator.predict_proba(Xdev.iloc[va])))
                folds.append(row)
                rows.append(row)
            frame = pd.DataFrame(rows)
            summary = dict(scenario=scenario, model=name)
            for metric in ["accuracy", "macro_f1", "weighted_f1", "fit_seconds"]:
                summary[f"{metric}_mean"] = frame[metric].mean()
                summary[f"{metric}_std"] = frame[metric].std(ddof=1)
            summaries.append(summary)
            print(f"  {name}: CV macro-F1={summary['macro_f1_mean']:.4f}", flush=True)
        # Candidate list, parameters and selection metric are fixed before test evaluation.
        winner = max([s for s in summaries if s["scenario"] == scenario], key=lambda s: s["macro_f1_mean"])["model"]
        fitted = clone(candidates[winner]).fit(Xdev, ydev)
        pred = np.asarray(fitted.predict(Xtest)).ravel().astype(int)
        proba = fitted.predict_proba(Xtest)
        tests.append(dict(scenario=scenario, model=winner, development_rows=len(train), test_rows=len(test),
                          **scores(ytest, pred, proba), **bootstrap_interval(ytest, pred)))
        pd.DataFrame({"row_index": test, "actual": ytest, "predicted": pred,
                      **{f"probability_{i}": proba[:, i] for i in range(len(labels))}}).to_csv(
                          REPORTS / f"{scenario}_predictions.csv", index=False)
        report = classification_report(ytest, pred, labels=np.arange(len(labels)), target_names=labels,
                                       output_dict=True, zero_division=0)
        (REPORTS / f"{scenario}_classification.json").write_text(json.dumps(report, indent=2))
        plot_confusion(scenario, winner, ytest, pred, labels)
        importance = permutation_importance(fitted, Xtest, ytest, scoring="f1_macro", n_repeats=15,
                                             random_state=SEED, n_jobs=1)
        imp = pd.DataFrame({"feature": X.columns, "mean": importance.importances_mean,
                            "std": importance.importances_std}).sort_values("mean", ascending=False)
        imp.to_csv(REPORTS / f"{scenario}_importance.csv", index=False)
        plot_importance(imp, scenario)
        if scenario == "breast_cancer":
            plot_cancer_curves(ytest, proba[:, 1], winner)
        # Descriptive sensitivity analysis on development data only; does not reselect the winner.
        for depth in [3, 6, 9]:
            candidate = clone(candidates["LightGBM"]).set_params(model__max_depth=depth)
            for fold, (tr, va) in enumerate(cv, 1):
                estimator = clone(candidate).fit(Xdev.iloc[tr], ydev[tr])
                depth_rows.append(dict(scenario=scenario, depth=depth, fold=fold,
                                       macro_f1=f1_score(ydev[va], estimator.predict(Xdev.iloc[va]), average="macro")))
        pd.DataFrame(folds).to_csv(REPORTS / "cv_folds.csv", index=False)
        pd.DataFrame(summaries).to_csv(REPORTS / "cv_summary.csv", index=False)
        pd.DataFrame(tests).to_csv(REPORTS / "test_summary.csv", index=False)

    depth_frame = pd.DataFrame(depth_rows)
    depth_frame.to_csv(REPORTS / "depth_folds.csv", index=False)
    (REPORTS / "splits.json").write_text(json.dumps(splits, indent=2))
    summary = pd.DataFrame(summaries)
    plot_comparison(summary, scenarios)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, scenario in zip(axes, scenarios):
        part = summary[summary.scenario == scenario].set_index("model").loc[NAMES]
        ax.barh(NAMES, part.fit_seconds_mean, color=COLORS)
        ax.set(title=scenario.replace("_", " ").title(), xlabel="Mean fit time per CV fold (seconds)")
        ax.invert_yaxis()
    save_figure(fig, "training_time")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, scenario in zip(axes, scenarios):
        part = depth_frame[depth_frame.scenario == scenario].groupby("depth").macro_f1.agg(["mean", "std"])
        ax.errorbar(part.index, part["mean"], yerr=part["std"], marker="o", color=COLORS[3], capsize=5)
        ax.set(title=scenario.replace("_", " ").title(), xlabel="LightGBM maximum depth", ylabel="Development macro-F1")
        ax.set_xticks([3, 6, 9])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    save_figure(fig, "depth_sensitivity")
    metadata = dict(seed=SEED, folds=5, test_fraction=.2, iterations=iterations, threads=4,
                    selection_metric="development CV mean macro-F1", python=platform.python_version(),
                    platform=platform.platform(), packages={p: version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "xgboost", "lightgbm", "catboost"]},
                    data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "data").glob("*.csv")})
    (REPORTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nFinished. Results and figures are in reports/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    run(args.iterations)
