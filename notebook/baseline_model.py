from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "src" / "Chocolate_Sales.csv"
RESULTS_PATH = ROOT_DIR / "results" / "metrics_table.csv"
TARGET_COLUMN = "Amount"
DROP_COLUMNS = ["Order_ID", "Order_Date"]
NUMERIC_COLUMNS = [
    "Discount_Pct",
    "Price_per_Box",
    "Marketing_Spend",
    "Boxes_Shipped",
    TARGET_COLUMN,
]


def clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        df[column] = clean_numeric(df[column])

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Order_Year"] = df["Order_Date"].dt.year
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_Day"] = df["Order_Date"].dt.day
    return df


def build_model(X):
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def train_baseline():
    df = load_data()
    df = df.dropna(subset=[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = build_model(X_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": "baseline_ridge_regression",
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    return model, metrics


def save_metrics(metrics, path=RESULTS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)


if __name__ == "__main__":
    _, baseline_metrics = train_baseline()
    save_metrics(baseline_metrics)

    print("Baseline model trained")
    for name, value in baseline_metrics.items():
        print(f"{name}: {value}")
