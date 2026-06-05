"""Tune LR and XGBoost on the QQQ excess-return target.

Target:
    excess_qqq_gt_0.2pct = 1 if NVDA daily return - QQQ daily return > 0.2%p.

Selection:
    Hyperparameters are selected only on the validation split by ROC-AUC.
    Thresholds are selected only on the validation split by balanced accuracy.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
TARGET_NAME = "excess_qqq_gt_0.2pct"
TARGET_THRESHOLD = 0.002
RESULTS_DIR = Path("reports/results")


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    features = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv("data/raw/prices_raw.csv", index_col=0, parse_dates=True)
    excess_return = prices["NVDA"].pct_change() - prices["QQQ"].pct_change()
    df = features.drop(columns=["target"]).join(excess_return.rename("target_excess"), how="inner")
    df = df.dropna()
    y = (df["target_excess"] > TARGET_THRESHOLD).astype(int)
    feature_cols = [c for c in features.columns if c != "target"]
    return df, y, feature_cols


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_score = 0.5, -1.0
    for threshold in np.arange(0.30, 0.71, 0.01):
        pred = (y_prob >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_t = threshold
            best_score = score
    return round(float(best_t), 2), round(float(best_score), 4)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred) * 100), 2),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pred_positive_rate": round(float(pred.mean()), 4),
    }


def add_common_trial_fields(
    row: dict,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    row.update(
        {
            "target": TARGET_NAME,
            "train_pos_rate": round(float(y_train.mean()), 4),
            "val_pos_rate": round(float(y_val.mean()), 4),
            "test_pos_rate": round(float(y_test.mean()), 4),
            "test_n": int(len(y_test)),
            "test_majority_baseline": round(
                float(max(y_test.mean(), 1 - y_test.mean()) * 100), 2
            ),
        }
    )
    return row


def tune_lr(X_train, y_train, X_val, y_val, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    rows = []
    configs = list(
        product(
            [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            ["l2"],
            [None, "balanced"],
        )
    )
    for C, penalty, class_weight in configs:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=C,
                        penalty=penalty,
                        class_weight=class_weight,
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val)[:, 1]
        threshold, val_bal_acc = best_threshold(y_val, prob_val)
        row = {
            "model": "LR",
            "C": C,
            "penalty": penalty,
            "class_weight": class_weight or "none",
            "val_threshold": threshold,
            "val_bal_acc_at_threshold": val_bal_acc,
            "val_roc_auc": round(float(roc_auc_score(y_val, prob_val)), 4),
        }
        row.update({f"val_{k}": v for k, v in classification_metrics(y_val, prob_val, threshold).items()})
        row = add_common_trial_fields(row, y_train, y_val, y_test)
        rows.append(row)

    trials = pd.DataFrame(rows).sort_values(
        ["val_roc_auc", "val_bal_acc_at_threshold"], ascending=False
    )
    best_params = trials.iloc[0].to_dict()
    best_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=best_params["C"],
                    penalty=best_params["penalty"],
                    class_weight=None
                    if best_params["class_weight"] == "none"
                    else best_params["class_weight"],
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    best_model.fit(X_train, y_train)
    prob_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, prob_test, best_params["val_threshold"])
    summary = {
        "model": "LR",
        "selection_metric": "val_roc_auc",
        "C": best_params["C"],
        "penalty": best_params["penalty"],
        "class_weight": best_params["class_weight"],
        "val_threshold": best_params["val_threshold"],
        "val_roc_auc": best_params["val_roc_auc"],
        "val_bal_acc_at_threshold": best_params["val_bal_acc_at_threshold"],
    }
    summary.update({f"test_{k}": v for k, v in test_metrics.items()})
    return trials, summary


def tune_xgb(X_train, y_train, X_val, y_val, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    rows = []
    configs = list(
        product(
            [1, 2, 3],
            [0.005, 0.01, 0.03, 0.05],
            [100, 200, 400],
            [3, 5, 10],
            [0.6, 0.8, 1.0],
            [0.6, 0.8, 1.0],
            [0.0, 0.1, 0.5],
            [1.0, 2.0, 5.0],
        )
    )
    for (
        max_depth,
        learning_rate,
        n_estimators,
        min_child_weight,
        subsample,
        colsample_bytree,
        reg_alpha,
        reg_lambda,
    ) in configs:
        model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val)[:, 1]
        threshold, val_bal_acc = best_threshold(y_val, prob_val)
        row = {
            "model": "XGBoost",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "val_threshold": threshold,
            "val_bal_acc_at_threshold": val_bal_acc,
            "val_roc_auc": round(float(roc_auc_score(y_val, prob_val)), 4),
        }
        row.update({f"val_{k}": v for k, v in classification_metrics(y_val, prob_val, threshold).items()})
        row = add_common_trial_fields(row, y_train, y_val, y_test)
        rows.append(row)

    trials = pd.DataFrame(rows).sort_values(
        ["val_roc_auc", "val_bal_acc_at_threshold"], ascending=False
    )
    best_params = trials.iloc[0].to_dict()
    best_model = XGBClassifier(
        max_depth=int(best_params["max_depth"]),
        learning_rate=best_params["learning_rate"],
        n_estimators=int(best_params["n_estimators"]),
        min_child_weight=best_params["min_child_weight"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
        verbosity=0,
    )
    best_model.fit(X_train, y_train)
    prob_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, prob_test, best_params["val_threshold"])
    summary = {
        "model": "XGBoost",
        "selection_metric": "val_roc_auc",
        "max_depth": int(best_params["max_depth"]),
        "learning_rate": best_params["learning_rate"],
        "n_estimators": int(best_params["n_estimators"]),
        "min_child_weight": best_params["min_child_weight"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "reg_alpha": best_params["reg_alpha"],
        "reg_lambda": best_params["reg_lambda"],
        "val_threshold": best_params["val_threshold"],
        "val_roc_auc": best_params["val_roc_auc"],
        "val_bal_acc_at_threshold": best_params["val_bal_acc_at_threshold"],
    }
    summary.update({f"test_{k}": v for k, v in test_metrics.items()})
    return trials, summary


def main() -> None:
    df, y, feature_cols = load_data()
    train_idx = df.index <= "2023-06-30"
    val_idx = (df.index >= "2023-07-01") & (df.index <= "2024-06-30")
    test_idx = df.index >= "2024-07-01"
    X_train = df.loc[train_idx, feature_cols].values
    X_val = df.loc[val_idx, feature_cols].values
    X_test = df.loc[test_idx, feature_cols].values
    y_train = y.loc[train_idx].values
    y_val = y.loc[val_idx].values
    y_test = y.loc[test_idx].values

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lr_trials, lr_summary = tune_lr(X_train, y_train, X_val, y_val, X_test, y_test)
    lr_trials.to_csv(RESULTS_DIR / "tuning_lr_excess_qqq_02_trials.csv", index=False)

    xgb_trials, xgb_summary = tune_xgb(X_train, y_train, X_val, y_val, X_test, y_test)
    xgb_trials.to_csv(RESULTS_DIR / "tuning_xgb_excess_qqq_02_trials.csv", index=False)

    summary = pd.DataFrame([lr_summary, xgb_summary]).sort_values(
        ["test_roc_auc", "test_balanced_accuracy"], ascending=False
    )
    summary.to_csv(RESULTS_DIR / "tuning_lr_xgb_excess_qqq_02_summary.csv", index=False)

    print("=== LR top 10 by validation ROC-AUC ===")
    print(lr_trials.head(10).to_string(index=False))
    print("\n=== XGBoost top 10 by validation ROC-AUC ===")
    print(xgb_trials.head(10).to_string(index=False))
    print("\n=== Test evaluation of validation-selected configs ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
