from benchmark import plot_comparison

cv = pd.read_csv(REPORTS / "cv_summary.csv")
scenario_order = ["obesity_full", "obesity_without_body_size", "breast_cancer"]
scenario_names = {
    "obesity_full": "Obesity · all inputs",
    "obesity_without_body_size": "Obesity · without height and weight",
    "breast_cancer": "Breast cancer",
}
view = cv[["scenario", "model", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "fit_seconds_mean"]].copy()
view["scenario"] = view["scenario"].map(scenario_names)
view = view.rename(columns={
    "scenario": "Scenario", "model": "Model", "macro_f1_mean": "Macro-F1",
    "macro_f1_std": "Fold SD", "accuracy_mean": "Accuracy", "fit_seconds_mean": "Fit time (s)",
})
for column in ["Macro-F1", "Fold SD", "Accuracy"]:
    view[column] = view[column].map(lambda value: f"{value:.2%}")
view["Fit time (s)"] = view["Fit time (s)"].map(lambda value: f"{value:.3f}")
display(view.set_index(["Scenario", "Model"]))

# Regenerate the image from the current CSV rather than displaying an old export.
plot_comparison(cv, scenario_order)
figure("model_comparison")
