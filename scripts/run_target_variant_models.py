"""Run target-variant experiments across classical and neural models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

SEED = 42
WINDOW = 20
EPOCHS = 80
BATCH_SIZE = 32
CALENDAR_COLS = [
    "is_nvda_post_earnings",
    "is_nvda_earnings_eve",
    "is_fomc_day",
    "is_cpi_day",
]


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class DilatedTCN(nn.Module):
    def __init__(self, input_dim: int, num_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for dilation in [1, 2, 4, 8]:
            layers += [
                nn.Conv1d(in_ch, num_channels, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = num_channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        return self.fc(x[:, :, -1]).squeeze(1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(self.dropout(output[:, -1, :])).squeeze(1)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(self.dropout(output[:, -1, :])).squeeze(1)


class InvertedTransformerClassifier(nn.Module):
    """iTransformer-style classifier: variables are tokens, time is embedded."""

    def __init__(
        self,
        window: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.time_projection = nn.Linear(window, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.time_projection(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(self.dropout(x)).squeeze(1)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def load_dataset() -> tuple[pd.DataFrame, dict[str, pd.Series], list[str], list[str]]:
    features = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv("data/raw/prices_raw.csv", index_col=0, parse_dates=True)
    returns = pd.DataFrame(
        {
            "nvda_return_target": prices["NVDA"].pct_change(),
            "qqq_return_target": prices["QQQ"].pct_change(),
        }
    )
    returns["excess_qqq_return_target"] = (
        returns["nvda_return_target"] - returns["qqq_return_target"]
    )
    df = features.drop(columns=["target"]).join(returns, how="inner").dropna()
    targets = {
        "current_direction": (df["nvda_return_target"] > 0).astype(int),
        "abs_return_gt_0.2pct": (df["nvda_return_target"] > 0.002).astype(int),
        "excess_qqq_gt_0": (df["excess_qqq_return_target"] > 0).astype(int),
        "excess_qqq_gt_0.2pct": (df["excess_qqq_return_target"] > 0.002).astype(int),
    }
    feature_cols = [c for c in features.columns if c != "target"]
    market_cols = [c for c in feature_cols if c not in CALENDAR_COLS]
    return df, targets, feature_cols, market_cols


def make_sequences(X: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(window, len(X)):
        xs.append(X[i - window : i])
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for threshold in np.arange(0.30, 0.71, 0.01):
        pred = (y_prob >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_s:
            best_s = score
            best_t = threshold
    return round(float(best_t), 2), round(float(best_s), 4)


def build_metrics(
    target_name: str,
    model_name: str,
    feature_set: str,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    prob_val: np.ndarray,
    prob_test: np.ndarray,
) -> dict:
    threshold, val_bal_acc = best_threshold(y_val, prob_val)
    pred = (prob_test >= threshold).astype(int)
    return {
        "target": target_name,
        "model": model_name,
        "feature_set": feature_set,
        "threshold_by_val_bal_acc": threshold,
        "val_bal_acc_at_threshold": val_bal_acc,
        "train_pos_rate": round(float(y_train.mean()), 4),
        "val_pos_rate": round(float(y_val.mean()), 4),
        "test_pos_rate": round(float(y_test.mean()), 4),
        "test_n": int(len(y_test)),
        "test_majority_baseline": round(float(max(y_test.mean(), 1 - y_test.mean()) * 100), 2),
        "test_accuracy": round(float(accuracy_score(y_test, pred) * 100), 2),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "test_precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "test_recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "test_f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "test_mcc": round(float(matthews_corrcoef(y_test, pred)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, prob_test)), 4),
        "pred_positive_rate": round(float(pred.mean()), 4),
    }


def fit_mlp(X_train, y_train, X_val, y_val, X_test) -> tuple[np.ndarray, np.ndarray]:
    set_seed()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(y_train.astype(float))),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    best_auc, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(model(torch.FloatTensor(Xva))).numpy()
        auc = roc_auc_score(y_val, prob_val)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(model(torch.FloatTensor(Xva))).numpy()
        prob_test = torch.sigmoid(model(torch.FloatTensor(Xte))).numpy()
    return prob_val, prob_test


def fit_tcn(X_train, y_train, X_val, y_val, X_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    Xtr_s, ytr_s = make_sequences(Xtr, y_train, WINDOW)
    Xva_s, yva_s = make_sequences(Xva, y_val, WINDOW)
    Xte_s, _ = make_sequences(Xte, np.zeros(len(Xte)), WINDOW)
    model = DilatedTCN(Xtr_s.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr_s), torch.FloatTensor(ytr_s.astype(float))),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    best_auc, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(model(torch.FloatTensor(Xva_s))).numpy()
        auc = roc_auc_score(yva_s, prob_val)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(model(torch.FloatTensor(Xva_s))).numpy()
        prob_test = torch.sigmoid(model(torch.FloatTensor(Xte_s))).numpy()
    return yva_s, prob_val, prob_test


def fit_lstm(X_train, y_train, X_val, y_val, X_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    Xtr_s, ytr_s = make_sequences(Xtr, y_train, WINDOW)
    Xva_s, yva_s = make_sequences(Xva, y_val, WINDOW)
    Xte_s, _ = make_sequences(Xte, np.zeros(len(Xte)), WINDOW)
    model = LSTMClassifier(Xtr_s.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr_s), torch.FloatTensor(ytr_s.astype(float))),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    best_auc, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(model(torch.FloatTensor(Xva_s))).numpy()
        auc = roc_auc_score(yva_s, prob_val)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(model(torch.FloatTensor(Xva_s))).numpy()
        prob_test = torch.sigmoid(model(torch.FloatTensor(Xte_s))).numpy()
    return yva_s, prob_val, prob_test


def fit_gru(X_train, y_train, X_val, y_val, X_test) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    Xtr_s, ytr_s = make_sequences(Xtr, y_train, WINDOW)
    Xva_s, yva_s = make_sequences(Xva, y_val, WINDOW)
    Xte_s, _ = make_sequences(Xte, np.zeros(len(Xte)), WINDOW)
    model = GRUClassifier(Xtr_s.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr_s), torch.FloatTensor(ytr_s.astype(float))),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    best_auc, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(model(torch.FloatTensor(Xva_s))).numpy()
        auc = roc_auc_score(yva_s, prob_val)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(model(torch.FloatTensor(Xva_s))).numpy()
        prob_test = torch.sigmoid(model(torch.FloatTensor(Xte_s))).numpy()
    return yva_s, prob_val, prob_test


def fit_itransformer(
    X_train, y_train, X_val, y_val, X_test
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    set_seed()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    Xtr_s, ytr_s = make_sequences(Xtr, y_train, WINDOW)
    Xva_s, yva_s = make_sequences(Xva, y_val, WINDOW)
    Xte_s, _ = make_sequences(Xte, np.zeros(len(Xte)), WINDOW)
    model = InvertedTransformerClassifier(window=WINDOW)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr_s), torch.FloatTensor(ytr_s.astype(float))),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    best_auc, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            prob_val = torch.sigmoid(model(torch.FloatTensor(Xva_s))).numpy()
        auc = roc_auc_score(yva_s, prob_val)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(model(torch.FloatTensor(Xva_s))).numpy()
        prob_test = torch.sigmoid(model(torch.FloatTensor(Xte_s))).numpy()
    return yva_s, prob_val, prob_test


def run_one(target_name: str, model_name: str) -> dict:
    set_seed()
    df, targets, feature_cols, market_cols = load_dataset()
    if target_name not in targets:
        raise ValueError(f"Unknown target: {target_name}")

    train_idx = df.index <= "2023-06-30"
    val_idx = (df.index >= "2023-07-01") & (df.index <= "2024-06-30")
    test_idx = df.index >= "2024-07-01"
    y = targets[target_name]
    y_train = y.loc[train_idx].values
    y_val = y.loc[val_idx].values
    y_test = y.loc[test_idx].values

    if model_name in {"TCN_redesign", "LSTM_redesign", "GRU_redesign", "iTransformer"}:
        X_train = df.loc[train_idx, market_cols].values
        X_val = df.loc[val_idx, market_cols].values
        X_test = df.loc[test_idx, market_cols].values
        if model_name == "TCN_redesign":
            y_val_eval, prob_val, prob_test = fit_tcn(X_train, y_train, X_val, y_val, X_test)
        elif model_name == "LSTM_redesign":
            y_val_eval, prob_val, prob_test = fit_lstm(X_train, y_train, X_val, y_val, X_test)
        elif model_name == "GRU_redesign":
            y_val_eval, prob_val, prob_test = fit_gru(X_train, y_train, X_val, y_val, X_test)
        else:
            y_val_eval, prob_val, prob_test = fit_itransformer(
                X_train, y_train, X_val, y_val, X_test
            )
        return build_metrics(
            target_name,
            model_name,
            "market_18",
            y_train[WINDOW:],
            y_val_eval,
            y_test[WINDOW:],
            prob_val,
            prob_test,
        )

    X_train = df.loc[train_idx, feature_cols].values
    X_val = df.loc[val_idx, feature_cols].values
    X_test = df.loc[test_idx, feature_cols].values
    if model_name == "LR":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.001, max_iter=1000, random_state=SEED)),
            ]
        )
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]
    elif model_name == "XGBoost":
        model = XGBClassifier(
            max_depth=2,
            learning_rate=0.01,
            n_estimators=200,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.5,
            n_jobs=1,
            random_state=SEED,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]
    elif model_name == "MLP":
        prob_val, prob_test = fit_mlp(X_train, y_train, X_val, y_val, X_test)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return build_metrics(target_name, model_name, "all_22", y_train, y_val, y_test, prob_val, prob_test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "LR",
            "XGBoost",
            "MLP",
            "TCN_redesign",
            "LSTM_redesign",
            "GRU_redesign",
            "iTransformer",
        ],
    )
    parser.add_argument("--output", default="reports/results/target_variant_all_models_metrics.csv")
    args = parser.parse_args()
    row = run_one(args.target, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        current = pd.read_csv(output)
        current = current[
            ~((current["target"] == row["target"]) & (current["model"] == row["model"]))
        ]
        out = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out = out.sort_values(["test_roc_auc", "test_balanced_accuracy"], ascending=False)
    out.to_csv(output, index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
