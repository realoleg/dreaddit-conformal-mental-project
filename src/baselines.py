from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import build_metrics_row

REQUIRED_COLUMNS = ["text", "label"]


def load_processed_split(path: str | Path) -> pd.DataFrame:

    """
    Load one processed CSV split and validate the minimum required columns.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")
    
    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")
    
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    if "example_id" not in df.columns:
        df.insert(0, "example_id", range(len(df)))

    return df


def build_baseline_models(
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1,2),
) -> dict[str, Pipeline]:
    
    """
    Create two baseline pipelines used further
    """

    vectorized_kwargs = {
        "max_features": max_features,
        "ngram_range": ngram_range,
        "lowercase": True,
        "strip_accents": "unicode",
    }

    models = {
        "tfidf_logreg": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**vectorized_kwargs)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**vectorized_kwargs)),
                (
                    "clf",
                    LinearSVC(
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models


def fit_models(
        models: dict[str, Pipeline],
        train_df: pd.DataFrame,
) -> dict[str, Pipeline]:
    
    """
    Fit each baseline model on training DataFrame
    """

    x_train = train_df["text"].tolist()
    y_train = train_df["label"].to_numpy()

    for model in models.values():
        model.fit(x_train, y_train)
    
    return models


def evaluate_models_on_split(
        models: dict[str, Pipeline],
        df: pd.DataFrame,
        split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Run predictions for all models on one split and return two df: metrics and predicitons
    """

    x = df["text"].tolist()
    y = df["label"].to_numpy()

    metrics_row: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    base_columns = ["example_id", "text", "label"]
    if "subreddit" in df.columns:
        base_columns.append("subreddit")
    
    for model_name, model in models.items():
        y_pred = model.predict(x).astype(int)

        metrics_row.append(
            build_metrics_row(
                model_name=model_name,
                split_name=split_name,
                y_true=y,
                y_pred=y_pred,
            )
        )

        pred_df = df[base_columns].copy()
        pred_df["split"] = split_name
        pred_df["model_name"] = model_name
        pred_df["pred_label"] = y_pred
        pred_df["correct"] = (pred_df["label"] == pred_df["pred_label"]).astype(int)

        prediction_frames.append(pred_df)
    
    metrics_df = pd.DataFrame(metrics_row)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    return metrics_df, predictions_df


def save_models(
        models: dict[str, Pipeline],
        output_dir: str | Path,
) -> None:
    
    """
    Save fitted baseline pipelines for potential reuse
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        save_path = output_path / f"{model_name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

