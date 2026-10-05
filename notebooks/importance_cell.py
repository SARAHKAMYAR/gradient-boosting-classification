from benchmark import plot_importance

for name in scenarios:
    importance = pd.read_csv(REPORTS / f"{name}_importance.csv")
    plot_importance(importance, name)
    figure(f"{name}_importance")
