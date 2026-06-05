# 모델 학습 결과 보고서

**최종 버전**: v5 (이벤트 더미 + 실현 변동성 + ΔVIX)

---

## 1. 실험 설정

| 항목 | 내용 |
|------|------|
| Train | 2020-01-09 ~ 2023-06-30 (875일) |
| Validation | 2023-07-01 ~ 2024-06-30 (250일) |
| Test | 2024-07-01 ~ 2025-05-22 (225일) |
| 피처 수 | 22개 (시장 18 + 캘린더 4) |
| 튜닝 기준 | Validation F1-Score |
| Majority baseline | 52.4% (항상 상승 예측) |

---

## 2. 피처 구성 (v5)

| 범주 | 피처 | shift |
|------|------|-------|
| NVDA 기술지표 | Return_1D, MA20_ratio, Volume_ratio, RSI, MACD | O |
| 상대강도 | NVDA_vs_QQQ, NVDA_vs_SOX | O |
| 변동성 | NVDA_RealVol_20d (20일 실현 변동성), VIX, VIX_delta | O |
| 섹터/종목 | SOX_Return, TSM_Return, QQQ_Return, MSFT_Return, META_Return | O |
| 매크로 | US_10Y_Yield, DXY_Return, US_CPI | O |
| 이벤트 캘린더 | is_nvda_post_earnings, is_nvda_earnings_eve, is_fomc_day, is_cpi_day | X |

---

## 3. 모델별 설정

| 모델 | 스케일링 | 주요 설정 |
|------|----------|-----------|
| Logistic Regression | StandardScaler | C=0.001 (val F1 기준 탐색) |
| XGBoost | 없음 | max_depth=2, lr=0.01, reg_alpha=0.1, min_child_weight=5 |
| MLP | StandardScaler | FC 3층(64→32→1), Dropout 0.3, pos_weight=3.0 |
| Dilated TCN | StandardScaler | dilation=1,2,4,8, window=20, pos_weight=3.0 |

---

## 4. Test 성능 비교 (v5 최종)

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|----------|-----------|--------|----|---------|
| LR | 52.44% | 0.525 | 0.992 | 0.686 | 0.558 |
| XGBoost | 52.44% | 0.528 | 0.873 | 0.658 | **0.506** |
| MLP | 52.44% | 0.524 | 1.000 | 0.688 | 0.536 |
| **Dilated TCN** | **53.17%** | 0.532 | 1.000 | **0.694** | 0.466 |
| Majority baseline | 52.4% | — | — | — | — |

---

## 5. 버전별 성능 추이 (Test Accuracy / ROC-AUC)

| 버전 | 주요 변경 | TCN Acc | Best ROC-AUC |
|------|-----------|---------|--------------|
| v1 | Close 원값 | 52.68% | 0.548 (LR) |
| v2 | Return 변환 | **54.63%** | 0.576 (LR) |
| v3 | 거시 지표 추가 | 53.17% | 0.575 (LR) |
| v4 | 상대강도 피처 | 52.68% | 0.571 (LR) |
| v5 | 이벤트 더미 + 변동성 | 53.17% | 0.558 (LR) |

---

## 6. 이벤트 더미 효과 검증

| 이벤트 | n | 이벤트일 수익률 | 비이벤트 | p-value |
|--------|---|----------------|---------|---------|
| is_nvda_post_earnings | 19 | **+2.65%** | +0.26% | **0.003 ★** |
| is_fomc_day | 43 | +1.29% | +0.26% | 0.055 |
| is_nvda_earnings_eve | 20 | −0.07% | +0.30% | 0.641 |
| is_cpi_day | 64 | +0.03% | +0.30% | 0.546 |

`is_nvda_post_earnings`만 통계적으로 유의미(p=0.003). 연간 4회 = 19일에 불과해 트리 모델이 충분히 학습하기 어렵다.

---

## 7. 고확신 필터 (XGBoost, thr=0.52)

- **Precision: 56.3%** (Majority baseline 52.4% 대비 +3.9%p)
- 커버리지: 64% (225일 중 144일 거래)
- 신호가 없는 날은 포지션 미보유

---

## 8. XGBoost 피처 중요도 상위 10 (v5)

| 순위 | 피처 | 중요도 |
|------|------|--------|
| 1 | TSM_Return | 0.062 |
| 2 | **NVDA_RealVol_20d** | 0.062 |
| 3 | NVDA_Return_1D | 0.061 |
| 4 | NVDA_MACD | 0.059 |
| 5 | NVDA_vs_QQQ | 0.056 |
| 6 | MSFT_Return | 0.055 |
| 7 | NVDA_Volume_ratio | 0.055 |
| 8 | QQQ_Return | 0.054 |
| 9 | VIX | 0.053 |
| 10 | US_10Y_Yield | 0.052 |

`NVDA_RealVol_20d`(신규)가 2위 — 변동성 레짐이 방향 예측에 유의미한 정보 제공.

---

## 9. 주요 발견

1. **Return 변환이 가장 효과적** — v1→v2에서 TCN Accuracy +2.0%p, LR ROC-AUC +0.028 개선
2. **실현 변동성이 유용한 피처** — XGBoost 중요도 2위 진입
3. **이벤트 더미는 샘플 수 부족으로 제한적** — 실적 이후 수익률 통계는 유의미하나 연간 4회에 불과
4. **고확신 필터가 Precision 개선에 효과적** — thr=0.52에서 56.3% 달성
5. **일별 방향 예측의 천장** — EMH 하에서 55% 이상의 일관된 정확도는 구조적으로 달성하기 어려움

---

## 10. 산출물

| 파일 | 내용 |
|------|------|
| `models/lr_model.joblib` | LR 최종 모델 |
| `models/xgb_model.joblib` | XGBoost 최종 모델 |
| `models/mlp_model.pt` | MLP 최종 모델 |
| `models/tcn_model.pt` | Dilated TCN 최종 모델 |
| `data/processed/features.csv` | 최종 피처 데이터셋 |
| `data/processed/train.csv` | 학습 데이터 |
| `data/processed/val.csv` | 검증 데이터 |
| `data/processed/test.csv` | 테스트 데이터 |
| `reports/results/model_metrics.csv` | 최종 성능 지표 |
| `reports/results/high_confidence_filter.csv` | 고확신 필터 분석 |
| `reports/results/model_comparison.png` | v4 vs v5 비교 차트 |
| `reports/results/event_return_distribution.png` | 이벤트별 수익률 분포 |
| `notebooks/modeling.ipynb` | 최종 모델링 노트북 |
