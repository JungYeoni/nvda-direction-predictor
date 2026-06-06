"""Predict excess return first, then classify by a validation-selected threshold.

Regression target:
    y_reg = NVDA daily return - QQQ daily return

Binary evaluation target:
    y_cls = 1 if y_reg > 0.2%p else 0

The model is fitted on the continuous excess return. The prediction threshold is
selected on the validation split by balanced accuracy, then applied once to test.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
TARGET_THRESHOLD = 0.002
RESULTS_DIR = Path("reports/results")


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    features = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv("data/raw/prices_raw.csv", index_col=0, parse_dates=True)
    excess_return = prices["NVDA"].pct_change() - prices["QQQ"].pct_change()
    df = features.drop(columns=["target"]).join(excess_return.rename("excess_return"), how="inner")
    df = df.dropna()
    y_reg = df["excess_return"]
    y_cls = (y_reg > TARGET_THRESHOLD).astype(int)
    feature_cols = [c for c in features.columns if c != "target"]
    return df, y_reg, y_cls, feature_cols


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(score, [0.05, 0.95])
    thresholds = np.linspace(low, high, 81)
    best_t, best_score = 0.0, -1.0
    for threshold in thresholds:
        pred = (score >= threshold).astype(int)
        metric = balanced_accuracy_score(y_true, pred)
        if metric > best_score:
            best_t = threshold
            best_score = metric
    return round(float(best_t), 6), round(float(best_score), 4)


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    corr = np.corrcoef(y_true, pred)[0, 1]
    return {
        "reg_mae": round(float(mean_absolute_error(y_true, pred)), 6),
        "reg_rmse": round(float(np.sqrt(mean_squared_error(y_true, pred))), 6),
        "reg_r2": round(float(r2_score(y_true, pred)), 4),
        "reg_corr": round(float(corr), 4) if np.isfinite(corr) else np.nan,
    }


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred) * 100), 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, score)), 4),
        "pred_positive_rate": round(float(pred.mean()), 4),
    }


def make_model(model_name: str, params: dict):
    if model_name == "Ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=params["alpha"], random_state=SEED)),
            ]
        )
    if model_name == "ElasticNet":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=params["alpha"],
                        l1_ratio=params["l1_ratio"],
                        max_iter=10000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if model_name == "Huber":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    HuberRegressor(
                        alpha=params["alpha"],
                        epsilon=params["epsilon"],
                        max_iter=1000,
                    ),
                ),
            ]
        )
    if model_name == "HistGBR":
        return HistGradientBoostingRegressor(
            max_iter=int(params["max_iter"]),
            learning_rate=params["learning_rate"],
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=params["l2_regularization"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=SEED,
        )
    if model_name == "XGBRegressor":
        return XGBRegressor(
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            n_estimators=int(params["n_estimators"]),
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
    if model_name == "LGBMRegressor":
        return LGBMRegressor(
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            n_estimators=int(params["n_estimators"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def param_grid() -> list[tuple[str, dict]]:
    grid: list[tuple[str, dict]] = []
    for alpha in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        grid.append(("Ridge", {"alpha": alpha}))
    for alpha, l1_ratio in product(
        [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03],
        [0.1, 0.3, 0.5, 0.7, 0.9],
    ):
        grid.append(("ElasticNet", {"alpha": alpha, "l1_ratio": l1_ratio}))
    for alpha, epsilon in product(
        [0.0001, 0.001, 0.01, 0.1, 1.0],
        [1.1, 1.35, 1.5, 1.75, 2.0],
    ):
        grid.append(("Huber", {"alpha": alpha, "epsilon": epsilon}))
    for max_iter, learning_rate, max_leaf_nodes, l2_regularization, min_samples_leaf in product(
        [50, 100, 200],
        [0.01, 0.03, 0.05],
        [3, 7, 15],
        [0.0, 0.1, 1.0],
        [20, 50, 100],
    ):
        grid.append(
            (
                "HistGBR",
                {
                    "max_iter": max_iter,
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                    "l2_regularization": l2_regularization,
                    "min_samples_leaf": min_samples_leaf,
                },
            )
        )
    for max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree in product(
        [1, 2, 3],
        [0.005, 0.01, 0.03],
        [100, 200],
        [3, 5, 10],
        [0.6, 0.8, 1.0],
        [0.6, 0.8, 1.0],
    ):
        grid.append(
            (
                "XGBRegressor",
                {
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "min_child_weight": min_child_weight,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "reg_alpha": 0.1,
                    "reg_lambda": 2.0,
                },
            )
        )
    for max_depth, learning_rate, n_estimators, num_leaves, min_child_samples, subsample, colsample_bytree in product(
        [3, 5, 7],
        [0.005, 0.01, 0.03],
        [100, 200],
        [15, 31, 63],
        [10, 20, 50],
        [0.6, 0.8, 1.0],
        [0.6, 0.8, 1.0],
    ):
        grid.append(
            (
                "LGBMRegressor",
                {
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "num_leaves": num_leaves,
                    "min_child_samples": min_child_samples,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "reg_alpha": 0.1,
                    "reg_lambda": 2.0,
                },
            )
        )
    return grid


def main() -> None:
    df, y_reg, y_cls, feature_cols = load_data()
    train_idx = df.index <= "2023-06-30"
    val_idx = (df.index >= "2023-07-01") & (df.index <= "2024-06-30")
    test_idx = df.index >= "2024-07-01"

    X_train = df.loc[train_idx, feature_cols]
    X_val = df.loc[val_idx, feature_cols]
    X_test = df.loc[test_idx, feature_cols]
    y_train_reg = y_reg.loc[train_idx].values
    y_val_reg = y_reg.loc[val_idx].values
    y_test_reg = y_reg.loc[test_idx].values
    y_val_cls = y_cls.loc[val_idx].values
    y_test_cls = y_cls.loc[test_idx].values

    rows = []
    for model_name, params in param_grid():
        model = make_model(model_name, params)
        model.fit(X_train, y_train_reg)
        pred_val_reg = model.predict(X_val)
        threshold, val_bal_acc = best_threshold(y_val_cls, pred_val_reg)
        row = {
            "model": model_name,
            **params,
            "val_threshold": threshold,
            "val_bal_acc_at_threshold": val_bal_acc,
            "val_roc_auc": round(float(roc_auc_score(y_val_cls, pred_val_reg)), 4),
            **{f"val_{k}": v for k, v in regression_metrics(y_val_reg, pred_val_reg).items()},
            **{
                f"val_cls_{k}": v
                for k, v in classification_metrics(y_val_cls, pred_val_reg, threshold).items()
            },
        }
        rows.append(row)

    trials = pd.DataFrame(rows).sort_values(
        ["val_roc_auc", "val_bal_acc_at_threshold"], ascending=False
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials.to_csv(RESULTS_DIR / "regress_then_classify_trials.csv", index=False)

    summaries = []
    for model_name, group in trials.groupby("model", sort=False):
        best = group.sort_values(["val_roc_auc", "val_bal_acc_at_threshold"], ascending=False).iloc[0]
        params = {
            col: best[col]
            for col in best.index
            if col
            not in {
                "model",
                "val_threshold",
                "val_bal_acc_at_threshold",
                "val_roc_auc",
            }
            and not col.startswith("val_")
            and pd.notna(best[col])
        }
        model = make_model(model_name, params)
        model.fit(X_train, y_train_reg)
        pred_test_reg = model.predict(X_test)
        cls_metrics = classification_metrics(y_test_cls, pred_test_reg, best["val_threshold"])
        reg_metrics = regression_metrics(y_test_reg, pred_test_reg)
        summary = {
            "model": model_name,
            "selection_metric": "val_roc_auc",
            "val_threshold": best["val_threshold"],
            "val_roc_auc": best["val_roc_auc"],
            "val_bal_acc_at_threshold": best["val_bal_acc_at_threshold"],
            "test_n": int(len(y_test_cls)),
            "test_pos_rate": round(float(y_test_cls.mean()), 4),
            "test_majority_baseline": round(
                float(max(y_test_cls.mean(), 1 - y_test_cls.mean()) * 100), 2
            ),
            **params,
            **{f"test_{k}": v for k, v in cls_metrics.items()},
            **{f"test_{k}": v for k, v in reg_metrics.items()},
        }
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["test_roc_auc", "test_balanced_accuracy"], ascending=False
    )
    summary_df.to_csv(RESULTS_DIR / "regress_then_classify_summary.csv", index=False)

    print("=== Top 15 regression-to-classification trials by validation ROC-AUC ===")
    print(trials.head(15).to_string(index=False))
    print("\n=== Test evaluation of validation-selected regressors ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
