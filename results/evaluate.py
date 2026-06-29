from pathlib import Path

import pandas as pd


RESULTS_PATH = Path(__file__).resolve().parent / "metrics_table.csv"


def load_metrics(path=RESULTS_PATH):
    """Load the saved model scores from the CSV file."""
    return pd.read_csv(path)


def print_model_comparison(metrics):
    """Print the most important evaluation results in simple terms."""
    display_columns = [
        "model",
        "mae",
        "rmse",
        "r2",
        "epsilon",
        "min_feature_epsilon",
        "max_feature_epsilon",
        "intercept_epsilon",
    ]
    available_columns = [column for column in display_columns if column in metrics.columns]

    print("Model comparison")
    print(metrics[available_columns].to_string(index=False))
    print()

    best_mae = metrics.loc[metrics["mae"].idxmin()]
    best_rmse = metrics.loc[metrics["rmse"].idxmin()]
    best_r2 = metrics.loc[metrics["r2"].idxmax()]

    print("Quick read")
    print(f"- Lowest average error: {best_mae['model']} (MAE {best_mae['mae']:.4f})")
    print(f"- Lowest large-error penalty: {best_rmse['model']} (RMSE {best_rmse['rmse']:.4f})")
    print(f"- Best explained variance: {best_r2['model']} (R2 {best_r2['r2']:.4f})")
    print()

    print("What the numbers mean")
    print("- MAE: average prediction error in sale amount units. Lower is better.")
    print("- RMSE: like MAE, but punishes large mistakes more. Lower is better.")
    print("- R2: how much target variation the model explains. Higher is better.")
    print("- epsilon: privacy budget. Lower means stronger privacy but usually more noise.")


if __name__ == "__main__":
    model_metrics = load_metrics()
    print_model_comparison(model_metrics)
