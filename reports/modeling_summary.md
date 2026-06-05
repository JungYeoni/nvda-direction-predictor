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
| 튜닝 기준 | Validation F1-Score (기존 최종 모델), 추가 TCN 재설계는 Validation ROC-AUC |
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
| Dilated TCN | 53.17% | 0.532 | 1.000 | 0.694 | 0.466 |
| Majority baseline | 52.4% | — | — | — | — |

ROC-AUC 기준으로는 Logistic Regression이 가장 안정적이다. Dilated TCN은 Accuracy와 F1이 높지만 `Recall=1.000`, `ROC-AUC=0.466`으로 상승 클래스에 치우친 퇴화 신호가 있어 최종 우위 모델로 해석하지 않는다. 또한 TCN은 `window=20` 때문에 test 앞 20일을 제외한 205일 기준으로 평가되며, 해당 aligned majority baseline은 53.17%다.

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

## 7. 고확신 필터 재검증 (XGBoost)

기존 `threshold=0.52`는 test 구간 precision을 보고 사후 선택한 성격이 있어 최종 성능으로 사용하지 않는다. Validation에서 threshold를 선택한 뒤 test에 1회 적용하는 방식으로 재검증했다.

| 선택 기준 | threshold | Val precision | Val coverage | Test precision | Test coverage |
|-----------|-----------|---------------|--------------|----------------|---------------|
| Val precision 최대, coverage ≥ 50% | 0.49 | 58.1% | 96.4% | 52.8% | 86.7% |

Validation 기준으로 선택하면 test precision 개선은 제한적이다. 따라서 고확신 필터는 보조 분석으로만 제시하고, 최종 모델 우위 근거로 사용하지 않는다.

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

## 9. TCN 재설계 실험

기존 TCN은 `pos_weight=3.0`, Validation F1 기준 best epoch 선택, 캘린더 dummy 포함으로 인해 상승 예측에 치우쳤다. 추가 실험에서는 best epoch를 Validation ROC-AUC로 선택하고, threshold는 Validation balanced accuracy 기준으로 정했다.

| TCN variant | Feature | pos_weight | Test n | Aligned baseline | Accuracy | Balanced Acc | F1 | MCC | ROC-AUC |
|-------------|---------|------------|--------|------------------|----------|--------------|----|-----|---------|
| 기존 v5 | 전체 22개 | 3.0 | 205 | 53.17% | 53.17% | — | 0.694 | — | 0.466 |
| 재설계 best | 시장 피처 18개 | 없음 | 205 | 53.17% | 54.15% | 0.538 | 0.580 | 0.076 | 0.522 |

재설계 TCN은 all-up 퇴화가 줄고 ROC-AUC가 0.522로 회복되었지만, LR의 ROC-AUC 0.558에는 미치지 못했다. 따라서 TCN은 추가 연구 후보로 남기되, 현재 최종 결론은 LR 중심으로 둔다.

---

## 10. 주요 발견

1. **Return 변환이 가장 효과적** — v1→v2에서 TCN Accuracy +2.0%p, LR ROC-AUC +0.028 개선
2. **실현 변동성이 유용한 피처** — XGBoost 중요도 2위 진입
3. **이벤트 더미는 샘플 수 부족으로 제한적** — 실적 이후 수익률 통계는 유의미하나 연간 4회에 불과
4. **고확신 필터의 실질 개선은 제한적** — validation 기준 threshold 선택 시 test precision 52.8%
5. **TCN은 재설계로 개선 가능하나 최종 우위는 아님** — 시장 피처만 사용하면 ROC-AUC 0.522로 회복되지만 LR보다 낮음
6. **일별 방향 예측의 천장** — EMH 하에서 55% 이상의 일관된 정확도는 구조적으로 달성하기 어려움

---

## 11. 추가 타깃 실험

같은 feature와 split에서 타깃만 바꾼 실험에서는 `NVDA 수익률 - QQQ 수익률 > 0.2%p` 타깃이 가장 좋은 결과를 보였다.

| 타깃 | 모델 | Accuracy | Balanced Acc | ROC-AUC | MCC |
|------|------|----------|--------------|---------|-----|
| 기존 방향: NVDA return > 0 | LR | 53.33% | 0.528 | 0.558 | 0.058 |
| **QQQ 초과: NVDA - QQQ > 0.2%p** | **LR** | **57.33%** | **0.574** | **0.610** | **0.148** |

이 결과는 절대 방향 예측보다 시장 대비 상대 강도 예측이 더 학습 가능할 수 있음을 보여준다. 단, 투자 해석은 "NVDA 상승 여부"가 아니라 "NVDA의 QQQ 대비 초과 성과 여부"에 가깝다.

---

## 12. 산출물

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
| `reports/results/high_confidence_filter_val_selected.csv` | validation 기준 고확신 필터 재검증 |
| `reports/results/tcn_redesign_metrics.csv` | TCN 재설계 비교 실험 |
| `reports/results/target_variant_metrics.csv` | 대체 타깃 비교 실험 |
| `reports/results/model_comparison.png` | v4 vs v5 비교 차트 |
| `reports/results/event_return_distribution.png` | 이벤트별 수익률 분포 |
| `notebooks/modeling.ipynb` | 최종 모델링 노트북 |
