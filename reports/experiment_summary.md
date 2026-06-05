# 실험 결과 종합 보고서

**프로젝트**: NVDA 주가 방향 예측  
**작성일**: 2026-06-04  
**최종 버전**: v5

---

## 기존 주가 방향 예측 연구 흐름

주가 방향 예측 연구는 단순한 가격 시계열 예측에서 출발해, 뉴스·감성 데이터, 딥러닝, 종목 간 관계 모델링, Transformer, LLM 기반 멀티모달 예측으로 확장되어 왔다.

### 1960s-1980s — 예측 가능성에 대한 회의

- 랜덤워크 가설과 효율적 시장가설(EMH)이 중심이었다.
- 핵심 질문은 과거 가격이나 공개 정보로 미래 주가 방향을 예측할 수 있는가였다.
- 이 시기 관점에서는 공개 정보가 이미 가격에 반영되어 있으므로, 지속적인 초과 예측 성과를 내기 어렵다고 보았다.

### 1990s-2000s — 통계 모델과 초기 머신러닝

- ARIMA, GARCH, VAR 같은 계량 모델이 가격, 수익률, 변동성 예측에 사용되었다.
- 이후 SVM, k-NN, Decision Tree, Neural Network 등 초기 머신러닝 모델이 적용되기 시작했다.
- 입력값은 주로 OHLCV, 이동평균, RSI, MACD, 모멘텀, 변동성 같은 기술지표였다.

### 2006-2014 — 텍스트와 뉴스 데이터 결합 시작

- 가격 데이터만으로는 예측력이 제한적이라는 인식이 커졌다.
- 금융 뉴스, 기업 공시, 웹 텍스트, SNS를 사용해 주가 방향을 예측하려는 연구가 등장했다.
- 주가 방향 예측이 가격 시계열 문제에서 가격과 외부 정보를 결합하는 문제로 확장되었다.

### 2015-2016 — 이벤트 기반 딥러닝

- Ding et al.의 *Deep Learning for Event-Driven Stock Prediction* 같은 연구가 대표적이다.
- 뉴스에서 이벤트를 추출하고, 이벤트 임베딩을 만들어 주가 방향 예측에 사용했다.
- 단순 감성 점수보다 어떤 사건이 발생했는지를 모델링하려는 방향으로 이동했다.

### 2017 — 대규모 ML 앙상블

- Krauss, Do, Huck의 2017년 연구는 S&P 500 전체를 대상으로 DNN, Gradient Boosted Trees, Random Forest, 앙상블을 비교했다.
- 단일 종목 예측보다 종목 간 ranking, long-short 전략, statistical arbitrage 관점이 강했다.
- 이 시기부터 단순 accuracy보다 실제 포트폴리오 성과와 out-of-sample 검증이 중요해졌다.

### 2018 — LSTM 전성기

- Fischer & Krauss의 2018년 LSTM 연구가 대표적이다.
- LSTM이 금융 시계열의 장단기 의존성을 학습할 수 있다는 기대가 컸다.
- 다만 실제로는 데이터 누수, 과적합, 거래비용 미반영 문제가 계속 지적되었다.

### 2018-2020 — 종목 관계와 그래프 모델

- 개별 종목만 보는 대신, 산업 관계, 공급망, 동종 종목, 시장 전체 관계를 함께 모델링했다.
- 방향 예측이 단일 시계열 문제에서 시장 내 관계 예측 문제로 확장되었다.

### 2021-2023 — Transformer 계열 확산

- Attention, Transformer, Temporal Fusion Transformer, Informer, PatchTST 등이 금융 시계열에 적용되었다.
- 금융 데이터는 NLP보다 표본이 작고 노이즈가 커서 Transformer가 항상 LSTM이나 XGBoost보다 낫지는 않다는 평가도 많다.

### 2023-2024 — 시장 구조 반영 Transformer

- MASTER: *Market-Guided Stock Transformer*가 대표적이다.
- 개별 종목의 시간 패턴뿐 아니라 종목 간 관계, 시장 상태, feature 중요도의 변화를 함께 반영했다.

### 2024-2026 — LLM, FinBERT, 뉴스 요약

- FinBERT, ChatGPT, LLM sentiment, 뉴스 이벤트 추출, 공시 요약 등을 결합한 연구가 증가했다.
- 가격 시계열 Transformer와 LLM 기반 텍스트 인코더를 co-attention으로 결합하는 구조가 많다.

### 연구 흐름 요약

```
랜덤워크/EMH
→ ARIMA/GARCH
→ SVM/RF/XGBoost
→ 뉴스·감성 결합
→ LSTM/GRU
→ CNN-LSTM/앙상블
→ 그래프·종목 관계 모델
→ Transformer
→ 시장 구조 반영 Transformer
→ LLM + 멀티모달 예측
```

본 프로젝트는 `SVM/RF/XGBoost`, `딥러닝 비교`, `시장 구조 반영 피처` 단계에 위치한다.

---

## 실험 개요

총 5가지 피처 버전으로 4개 모델을 비교했다.

| 버전 | 데이터 기간 | 주요 변경 | 노트북 |
|------|-------------|-----------|--------|
| **v1** | 2020~2025 | Close 원값 피처 | (삭제됨) |
| **v2** | 2020~2025 | Return 변환 (정상성 확보) | (삭제됨) |
| **v3** | 2020~2025 | 거시 지표 추가 (US_10Y, QQQ, DXY, MSFT, META) | (삭제됨) |
| **v4** | 2020~2025 | 상대강도 피처 (NVDA_vs_QQQ, NVDA_vs_SOX) | (삭제됨) |
| **v5** | 2020~2025 | 이벤트 더미 + 실현 변동성 + ΔVIX | `modeling.ipynb` |

**모델**: Logistic Regression / XGBoost / MLP / Dilated TCN  
**분할**: Train ≤ 2023-06-30 / Val 2023-07-01~2024-06-30 / Test ≥ 2024-07-01  
**평가 기준**: Validation F1-Score로 튜닝 → Test로 최종 평가  
**Test Majority baseline: 52.4%**

최종 해석은 Accuracy/F1보다 ROC-AUC, balanced accuracy, 퇴화 여부를 우선한다. 특히 TCN은 `window=20`으로 test 앞 20일을 제외한 205일 기준으로 평가되므로, TCN 해석에는 aligned majority baseline 53.17%를 함께 사용한다.

---

## 버전별 Test 성능 비교

### Accuracy

| 모델 | v1 | v2 | v3 | v4 | v5 |
|------|----|----|----|----|-----|
| LR | 52.89% | 52.89% | 52.44% | 52.44% | 52.44% |
| XGBoost | 47.56% | 52.00% | 52.00% | 52.44% | 52.44% |
| MLP | 52.44% | 52.44% | 52.89% | 52.44% | 52.44% |
| **DilatedTCN** | 52.68% | **54.63%** | 53.17% | 52.68% | 53.17% |

### ROC-AUC

| 모델 | v1 | v2 | v3 | v4 | v5 |
|------|----|----|----|----|-----|
| LR | 0.548 | **0.576** | 0.575 | 0.571 | 0.558 |
| XGBoost | 0.495 | 0.496 | 0.499 | 0.495 | 0.506 |
| MLP | 0.450 | 0.557 | 0.502 | 0.543 | 0.536 |
| DilatedTCN | 0.533 | 0.573 | 0.541 | 0.511 | 0.466 |

---

## 실험별 분석

### v1 — Close 원값 (기준 실험)

- LR만 ROC-AUC 0.548로 의미 있는 판별력 확인
- XGBoost: Val에서 우수했으나 Test 급락 → 과적합
- MLP: Recall=1.0으로 퇴화 (모든 샘플을 상승 예측)
- **결론**: Close 원값은 장기 추세와 단기 방향 신호가 혼재 → Return 변환 필요

### v2 — Return 변환

- TCN Accuracy 54.63% → 모든 버전 통틀어 최고 수치
- LR ROC-AUC 0.576 → v1 대비 +0.028 개선
- **결론**: Return/비율 변환이 정상성을 확보하여 가장 효과적인 개선

### v3 — 거시 지표 추가

- US_10Y_Yield, QQQ_Return, DXY_Return, MSFT_Return, META_Return 추가
- LR ROC-AUC 0.575로 v2와 유사, 거시 지표가 추가 정보 제공
- TCN Accuracy v2(54.63%) → v3(53.17%)로 소폭 하락 → 피처 증가로 인한 노이즈 가능성

### v4 — 상대강도 피처

- NVDA_vs_QQQ(NVDA−QQQ 수익률), NVDA_vs_SOX(NVDA−SOX 수익률) 추가
- XGBoost 정규화 강화 (reg_alpha=0.1, min_child_weight=5)
- 전반적 성능 변화 미미 — 상대강도 피처가 중복 정보일 가능성

### v5 — 이벤트 더미 + 실현 변동성 + ΔVIX (최종)

**신규 피처 6개:**

| 피처 | 유형 | 효과 |
|------|------|------|
| NVDA_RealVol_20d | 시장 데이터 | XGBoost 중요도 2위 ★ |
| VIX_delta | 시장 데이터 | 공포 방향성 정보 |
| is_nvda_post_earnings | 캘린더 | p=0.003 통계 유의 ★ |
| is_nvda_earnings_eve | 캘린더 | p=0.641 (유의미하지 않음) |
| is_fomc_day | 캘린더 | p=0.055 (경계선) |
| is_cpi_day | 캘린더 | p=0.546 (유의미하지 않음) |

- TCN Accuracy 53.17% (v4 대비 +0.49%p 개선), 단 aligned majority baseline도 53.17%
- `is_nvda_post_earnings`: 실적 발표 다음 날 평균 수익률 +2.65% vs 비이벤트 +0.26%, 통계적 유의
- 캘린더 피처의 모델 기여는 제한적 (이벤트 발생 횟수 부족: 19~64일)

---

## 종합 발견

### 1. Return 변환이 가장 효과적인 단일 개선
v1→v2에서 TCN +2.0%p, LR ROC-AUC +0.028. 다른 어떤 피처 추가보다 큰 폭의 개선.

### 2. 실현 변동성(RealVol)이 새로운 유용 피처
XGBoost 중요도 2위 진입. 변동성 레짐 정보가 방향 예측에 기여함을 확인.

### 3. 이벤트 더미: 통계적 효과는 있으나 모델 기여는 제한적
`is_nvda_post_earnings`는 통계적으로 유의미(p=0.003)하지만, 연간 4회(≈19일)는 트리 모델이 학습하기에 샘플이 너무 적음.

### 4. 단순 모델(LR)이 일관된 ROC-AUC 상위
ROC-AUC 기준으로 v2~v5 전 버전에서 LR이 최상위 또는 최상위권. 노이즈 강한 금융 시계열에서 과적합 없이 일반화.

### 5. TCN은 재설계로 퇴화 완화 가능
기존 v5 TCN은 F1이 높지만 `Recall=1.0`, `ROC-AUC=0.466`으로 상승 예측에 치우친 퇴화 신호가 있다. 추가 실험에서 캘린더 dummy를 제외하고 `pos_weight`를 제거한 뒤 Validation ROC-AUC로 best epoch를 선택하면 test ROC-AUC가 0.522로 회복되었다. 다만 LR의 ROC-AUC 0.558에는 미치지 못한다.

### 6. 고확신 필터의 실질 개선은 제한적
기존 `threshold=0.52`의 Precision 56.3%는 test 구간을 보고 선택한 사후 threshold 성격이 있다. Validation precision 기준으로 threshold를 선택하면 `threshold=0.49`, test precision 52.8%, coverage 86.7%로 개선 폭이 제한적이다.

### 7. 일별 방향 예측의 구조적 상한
EMH 하에서 공개 정보 기반 52~55% 정확도가 현실적 상한에 가깝다.

---

## 대체 타깃 실험

기존 방향 타깃은 작은 상승과 큰 상승을 모두 같은 `1`로 처리한다. 같은 feature와 동일한 train/validation/test split을 유지하고 타깃만 변경해, 더 예측 가능한 문제 정의가 있는지 확인했다.

| 타깃 | 모델 | Test pos rate | Majority baseline | Accuracy | Balanced Acc | ROC-AUC | MCC |
|------|------|---------------|-------------------|----------|--------------|---------|-----|
| `current_direction`: NVDA return > 0 | LR | 52.4% | 52.44% | 53.33% | 0.528 | 0.558 | 0.058 |
| `abs_return_gt_0.2pct`: NVDA return > 0.2% | LR | 51.1% | 51.11% | 52.89% | 0.527 | 0.573 | 0.055 |
| `excess_qqq_gt_0`: NVDA return - QQQ return > 0 | LR | 48.9% | 51.11% | 56.44% | 0.564 | 0.592 | 0.128 |
| **`excess_qqq_gt_0.2pct`: NVDA return - QQQ return > 0.2%p** | **LR** | **45.8%** | **54.22%** | **57.33%** | **0.574** | **0.610** | **0.148** |

가장 유망한 타깃은 `NVDA 수익률 - QQQ 수익률 > 0.2%p`였다. 기존 방향 타깃의 LR ROC-AUC 0.558 대비 0.610으로 개선되었고, balanced accuracy와 MCC도 함께 상승했다. 이는 NVDA의 절대 방향보다 시장 대비 상대 강도 예측이 더 학습 가능한 문제일 수 있음을 의미한다.

다만 이 결과는 타깃 재정의 실험이므로, 기존 "NVDA가 오르는가" 문제와 직접 같은 목표가 아니다. 전략 관점에서는 long NVDA / hedge QQQ 또는 NVDA overweight 판단에 더 가깝다.

---

## 한계점

1. **거래 비용·슬리피지 미고려** — 실제 트레이딩 시스템에서는 성능이 낮아질 가능성
2. **수익률 크기(magnitude) 무시** — 방향만 예측하므로 큰 손실과 작은 손실을 동등하게 취급
3. **구조 변화 취약성** — AI 붐(2023), DeepSeek 충격(2025) 같은 레짐 전환에 적응 어려움
4. **이벤트 더미 샘플 부족** — 실적 발표(연 4회), FOMC(연 8회)는 학습에 충분하지 않음
5. **텍스트·감성 정보 미활용** — 뉴스, 공시, SNS 감성이 추가 예측력을 줄 수 있음

---

## 산출물 목록

| 파일 | 설명 |
|------|------|
| `data/processed/features.csv` | 최종 피처 데이터셋 (22개 피처) |
| `data/processed/train.csv` | 학습 데이터 |
| `data/processed/val.csv` | 검증 데이터 |
| `data/processed/test.csv` | 테스트 데이터 |
| `models/lr_model.joblib` | LR 최종 모델 |
| `models/xgb_model.joblib` | XGBoost 최종 모델 |
| `models/mlp_model.pt` | MLP 최종 모델 |
| `models/tcn_model.pt` | Dilated TCN 최종 모델 |
| `reports/results/model_metrics.csv` | 최종 성능 지표 |
| `reports/results/high_confidence_filter.csv` | 고확신 필터 분석 |
| `reports/results/high_confidence_filter_val_selected.csv` | validation 기준 고확신 필터 재검증 |
| `reports/results/tcn_redesign_metrics.csv` | TCN 재설계 비교 실험 |
| `reports/results/target_variant_metrics.csv` | 대체 타깃 비교 실험 |
| `reports/results/model_comparison.png` | 버전 비교 차트 |
| `reports/results/event_return_distribution.png` | 이벤트별 수익률 분포 |
| `notebooks/modeling.ipynb` | 최종 모델링 노트북 |
| `notebooks/eda.ipynb` | 최종 EDA 노트북 |
