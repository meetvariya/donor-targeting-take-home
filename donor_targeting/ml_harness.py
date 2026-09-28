"""A small modelling harness: give it a feature frame X and a label frame y, get back a fitted
model, holdout metrics and holdout predictions.

    from donor_targeting.ml_harness import run

    result = run(X, y)                    # random stratified 75/25 split
    result = run(X, y, test_mask=is_aug)  # or your own (e.g. out-of-time) split
    print(result.summary())

NULL features are median-imputed and get a was-missing indicator, so a NULL
`centroid_cosine_similarity` ("no engagement in the last month") is a signal, not an error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

TOP_FRACTIONS = (0.25, 0.5, 0.75)


@dataclass
class HarnessResult:
    model: Pipeline
    metrics: dict[str, float]
    predictions: pd.DataFrame  # holdout rows: y_true, score

    def summary(self) -> str:
        return "\n".join(f"{k:>22}: {v:,.4f}" for k, v in self.metrics.items())


def default_model() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def precision_recall_at(y_true: np.ndarray, score: np.ndarray, frac: float) -> tuple[float, float]:
    """Precision and recall when only the top `frac` of rows by score are selected."""
    k = int(round(frac * len(score)))
    hits = y_true[np.argsort(-score, kind="stable")[:k]].sum()
    return hits / max(k, 1), hits / max(y_true.sum(), 1)


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    metrics = {
        "n_test": float(len(y_true)),
        "base_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
    }
    for frac in TOP_FRACTIONS:
        precision, recall = precision_recall_at(y_true, score, frac)
        metrics[f"precision@top{frac:.0%}"] = float(precision)
        metrics[f"recall@top{frac:.0%}"] = float(recall)
    return metrics


def _label(y: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"y must have exactly one column, got {list(y.columns)}")
        y = y.iloc[:, 0]
    return y.astype(int)


def run(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    *,
    test_mask: pd.Series | np.ndarray | None = None,
    model: Pipeline | None = None,
    test_size: float = 0.25,
    random_state: int = 0,
) -> HarnessResult:
    """Fit `model` (default: imputer + logistic regression) on the training rows and score the
    holdout. `test_mask` marks holdout rows; without it the split is random and stratified."""
    y = _label(y)
    if test_mask is None:
        _, test_rows = train_test_split(
            np.arange(len(y)), test_size=test_size, stratify=y, random_state=random_state
        )
        test = np.zeros(len(y), dtype=bool)
        test[test_rows] = True
    else:
        test = np.asarray(test_mask, dtype=bool)
    X_tr, X_te, y_tr, y_te = X[~test], X[test], y[~test], y[test]
    model = model if model is not None else default_model()
    model.fit(X_tr, y_tr)
    score = model.predict_proba(X_te)[:, 1]
    metrics = {"n_train": float(len(X_tr))} | evaluate(y_te.to_numpy(), score)
    predictions = pd.DataFrame({"y_true": y_te.to_numpy(), "score": score}, index=X_te.index)
    return HarnessResult(model=model, metrics=metrics, predictions=predictions)
