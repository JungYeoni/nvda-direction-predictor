"""Huber 회귀 예측값의 threshold별 precision/coverage 트레이드오프 분석."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = Path("reports/results")
FIGURES_DIR = Path("reports/figures")

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
preds = pd.read_csv(RESULTS_DIR / "final_assignment_predictions.csv", index_col=0, parse_dates=True)
y_true = preds["y_true"].values
score  = preds["predicted_excess_return"].values  # 연속 회귀 예측값

n_total = len(y_true)
majority_baseline = max(y_true.mean(), 1 - y_true.mean())

# ── threshold 범위: 예측값 분포 기준 ─────────────────────────────────────────
thresholds = np.linspace(np.quantile(score, 0.05), np.quantile(score, 0.95), 200)

rows = []
for thr in thresholds:
    mask = score >= thr
    n_trade = mask.sum()
    if n_trade == 0:
        continue
    y_sub  = y_true[mask]
    rows.append({
        "threshold":       round(float(thr), 6),
        "n_trades":        int(n_trade),
        "coverage_pct":    round(n_trade / n_total * 100, 1),
        "precision":       round(float(precision_score(y_sub, np.ones(n_trade, dtype=int), zero_division=0)), 4),
        "recall":          round(float(recall_score(y_true, mask.astype(int), zero_division=0)), 4),
        "accuracy":        round(float(accuracy_score(y_sub, np.ones(n_trade, dtype=int)) * 100), 2),
        "balanced_acc":    round(float(balanced_accuracy_score(y_true, mask.astype(int))), 4),
        "mcc":             round(float(matthews_corrcoef(y_true, mask.astype(int))), 4),
    })

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "threshold_sensitivity.csv", index=False)

# ── val 기준 선택 threshold 표시 ──────────────────────────────────────────────
summary = pd.read_csv(RESULTS_DIR / "regress_then_classify_summary.csv")
val_thr = float(summary[summary["model"] == "Huber"]["val_threshold"].iloc[0])

# ── 주요 지점 레이블 ──────────────────────────────────────────────────────────
label_points = {
    "전체 거래\n(thr=최솟값)":   df.iloc[0],
    "Val 선택\n(thr={:.4f})".format(val_thr): df.iloc[(df["threshold"] - val_thr).abs().argmin()],
    "최대 Precision":            df.loc[df["precision"].idxmax()],
    "최대 MCC":                  df.loc[df["mcc"].idxmax()],
}

# ── 시각화 ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Huber 회귀→분류: Threshold 민감도 분석\n(Test Set, 2024.07 ~ 2025.05, n=225)",
             fontsize=13, y=1.01)

palette = {"전체 거래\n(thr=최솟값)": "#4C72B0",
           "Val 선택\n(thr={:.4f})".format(val_thr): "#2ca02c",
           "최대 Precision": "#d62728",
           "최대 MCC": "#9467bd"}

# ① Precision vs Coverage
ax = axes[0, 0]
ax.plot(df["coverage_pct"], df["precision"], color="#333333", linewidth=1.8)
ax.axhline(majority_baseline, color="gray", linestyle="--", linewidth=1, label=f"Majority baseline ({majority_baseline*100:.1f}%)")
for label, row in label_points.items():
    ax.scatter(row["coverage_pct"], row["precision"], s=70, zorder=5, color=palette[label], label=label)
ax.set_xlabel("커버리지 (%)")
ax.set_ylabel("Precision")
ax.set_title("① Precision vs 커버리지")
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3)

# ② MCC vs Coverage
ax = axes[0, 1]
ax.plot(df["coverage_pct"], df["mcc"], color="#333333", linewidth=1.8)
ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="MCC=0 (랜덤)")
for label, row in label_points.items():
    ax.scatter(row["coverage_pct"], row["mcc"], s=70, zorder=5, color=palette[label], label=label)
ax.set_xlabel("커버리지 (%)")
ax.set_ylabel("MCC")
ax.set_title("② MCC vs 커버리지")
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3)

# ③ Precision & Recall vs Threshold
ax = axes[1, 0]
ax.plot(df["threshold"] * 100, df["precision"], label="Precision", color="#d62728", linewidth=1.8)
ax.plot(df["threshold"] * 100, df["recall"],    label="Recall",    color="#1f77b4", linewidth=1.8)
ax.axvline(val_thr * 100, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"Val threshold ({val_thr*100:.3f}%p)")
ax.set_xlabel("Threshold (초과수익률 %p)")
ax.set_ylabel("Score")
ax.set_title("③ Precision / Recall vs Threshold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# ④ 거래 횟수 vs Threshold
ax = axes[1, 1]
ax.bar(df["threshold"] * 100, df["n_trades"], width=(df["threshold"].max() - df["threshold"].min()) / len(df) * 100,
       color="#4C72B0", alpha=0.7)
ax.axvline(val_thr * 100, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"Val threshold ({val_thr*100:.3f}%p)")
ax.set_xlabel("Threshold (초과수익률 %p)")
ax.set_ylabel("거래 횟수 (일)")
ax.set_title("④ 거래 횟수 vs Threshold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "threshold_sensitivity.png", dpi=150, bbox_inches="tight")
print(f"차트 저장: {FIGURES_DIR}/threshold_sensitivity.png")

# ── 주요 지점 출력 ────────────────────────────────────────────────────────────
print("\n=== 주요 Threshold 비교 ===")
key_rows = []
for label, row in label_points.items():
    key_rows.append({
        "구분":        label.replace("\n", " "),
        "threshold":   f"{row['threshold']*100:.3f}%p",
        "n_trades":    int(row["n_trades"]),
        "coverage":    f"{row['coverage_pct']}%",
        "precision":   row["precision"],
        "recall":      row["recall"],
        "mcc":         row["mcc"],
    })
print(pd.DataFrame(key_rows).to_string(index=False))

print("\n=== Precision ≥ 0.60 구간 ===")
high_prec = df[df["precision"] >= 0.60][["threshold","n_trades","coverage_pct","precision","recall","mcc"]]
if len(high_prec):
    high_prec = high_prec.copy()
    high_prec["threshold"] = (high_prec["threshold"] * 100).round(3)
    print(high_prec.to_string(index=False))
else:
    print("없음")
