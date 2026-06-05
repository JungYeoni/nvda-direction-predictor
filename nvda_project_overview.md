# 과제 5 — NVIDIA(NVDA) 주가 방향 예측 모형

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 예측 종목 | NVIDIA Corporation (NVDA, NASDAQ) |
| 예측 대상 | 당일 종가의 방향 (오름: 1 / 내림: 0) |
| 문제 유형 | 이진 분류 (Binary Classification) |
| 데이터 기간 | 2020.01.02 ~ 2025.05.22 |
| 사용 모형 | Baseline 3종 + ML/DL 4종 + Weighted Ensemble |

---

## 2. 예측 문제 정의

### 타겟 변수

```
y_t = 1  if NVDA_Close_t > NVDA_Close_{t-1}   (오늘 상승)
y_t = 0  if NVDA_Close_t <= NVDA_Close_{t-1}  (오늘 하락/보합)
```

### 예측 시점 및 피처 사용 기준

```
예측 시점: t-1일 장 마감 후 → t일 개장 전
사용 정보: X_t = t-1일까지 확인 가능한 데이터 (shift(1) 적용)
```

t일 데이터는 예측 시점에 알 수 없으므로 모든 피처에 `shift(1)`을 적용한다.

---

## 3. 종목 선택 근거

NVIDIA는 세 가지 정체성이 주가에 동시에 작용한다.

- **AI 인프라 기업**: ChatGPT 출시(2022.11) 이후 데이터센터용 GPU 수요 폭발, AI 붐 최대 수혜주
- **반도체 기업**: 필라델피아 반도체 지수(SOX)와 높은 상관관계, TSMC 등 파운드리 파트너와 공급망 연결
- **미국 성장주**: 금리 변화, 달러 강세/약세, 시장 심리(VIX)에 민감한 고밸류에이션 성장주

이 세 가지 정체성을 반영하여 피처를 설계한다.

---

## 4. 데이터 기간 및 검증 전략

### Walk-forward Validation (3-fold)

단일 Train/Val/Test 분리 대신, 3개 구간을 순차적으로 검증한다.
각 폴드마다 학습 기간을 확장하여 특정 테스트 기간에 대한 의존도를 줄인다.

```
Fold 1: Train(2020~2022) ──→ Test(2023)
Fold 2: Train(2020~2023) ──→ Test(2024)
Fold 3: Train(2020~2024) ──→ Test(2025 YTD)
```

**최종 성능**: 3개 폴드 Accuracy / ROC-AUC의 평균 ± 표준편차로 보고

---

## 5. 피처 설계 (총 11개)

### 선택 원칙

- 가격 원값(Close) 대신 **수익률·상대 강도·비율** 중심으로 설계
- 상관계수 0.9 이상인 중복 피처 제거 (US_CPI ↔ US_10Y_Yield: 0.966)
- NVDA의 세 가지 정체성 카테고리 기준으로 구성

### 피처 목록

| # | 피처명 | 계산 방법 | 카테고리 |
|---|--------|-----------|----------|
| 1 | `NVDA_Return_1D` | `Close_t / Close_{t-1} - 1` | NVDA 기술 |
| 2 | `NVDA_MA20_ratio` | `(Close - MA20) / MA20` | NVDA 기술 |
| 3 | `NVDA_Volume_ratio` | `Volume / Volume_MA20` | NVDA 기술 |
| 4 | `NVDA_RSI` | RSI(14) | NVDA 기술 |
| 5 | `NVDA_MACD` | EMA(12) - EMA(26) | NVDA 기술 |
| 6 | `SOX_Return` | SOX 일별 수익률 | 반도체 섹터 |
| 7 | `TSM_Return` | TSM 일별 수익률 | 반도체 섹터 |
| 8 | `QQQ_Return` | QQQ 일별 수익률 | 거시경제 |
| 9 | `DXY_Return` | 달러 인덱스 일별 변화율 | 거시경제 |
| 10 | `US_10Y_Yield` | 미국 10년물 국채 금리 (레벨) | 거시경제 |
| 11 | `VIX` | CBOE 변동성 지수 (레벨) | 투자 심리 |

> **제거된 피처**: US_CPI (US_10Y_Yield와 r=0.966), MSFT_Return / META_Return (QQQ_Return과 r>0.87, ablation 효과 없음)

---

## 6. 모델 구성

### Baseline (비교 기준선 3종)

| 모델 | 예측 방법 |
|------|-----------|
| **Majority Class** | 항상 상승(1) 예측 — 가장 단순한 기준 |
| **Yesterday Direction** | 전일 방향을 그대로 예측 |
| **MA Direction** | 단기 MA > 장기 MA이면 상승 예측 (lookback 탐색) |

### ML / DL 모델 (4종)

| 모델 | 유형 | 주요 설정 |
|------|------|-----------|
| **Logistic Regression** | 통계 | C 그리드 탐색, StandardScaler |
| **XGBoost** | 트리 앙상블 | max_depth=3, reg_alpha=0.1, reg_lambda=1.5, min_child_weight=5 |
| **MLP** | 신경망 | FC 3층, Dropout, pos_weight=3, StandardScaler |
| **Dilated TCN** | 시계열 신경망 | dilation=1/2/4/8, Best Lookback 탐색, pos_weight=3 |

### Best Lookback 탐색

TCN과 MA Direction은 lookback 윈도우 크기를 탐색한다.

| 탐색 범위 | 기준 |
|-----------|------|
| 10, 14, 20, 30, 45일 | Val ROC-AUC 최대 |

### Ensemble

| 방법 | 설명 |
|------|------|
| **Weighted Ensemble** | Val ROC-AUC 최적화 가중치로 4개 모델 확률 결합 |

---

## 7. XAI — 예측 근거 분석

SHAP(SHapley Additive exPlanations)으로 모델 예측의 피처별 기여도를 분석한다.

| 대상 | 방법 |
|------|------|
| Logistic Regression | LinearExplainer |
| XGBoost | TreeExplainer |

**산출 그래프**:
- SHAP Summary Plot (전체 피처 중요도)
- SHAP Waterfall Plot (개별 예측 설명)

---

## 8. 평가 지표

| 지표 | 선택 이유 |
|------|-----------|
| **Accuracy** | 전체 방향 정확도 |
| **Precision** | 상승 신호의 정밀도 |
| **Recall** | 실제 상승일 포착률 |
| **F1-Score** | Precision-Recall 균형 |
| **ROC-AUC** | 임계값 무관 판별력 (주요 기준) |

최종 보고 형식:

| Model | Best Lookback | Accuracy | F1 | ROC-AUC |
|-------|--------------|----------|----|---------|
| Majority Class | — | | | |
| Yesterday Direction | — | | | |
| MA Direction | 탐색 | | | |
| LR | — | | | |
| XGBoost | — | | | |
| MLP | — | | | |
| **Dilated TCN** | **탐색** | | | |
| **Weighted Ensemble** | — | | | |

---

## 9. 노트북 구성

| 노트북 | 내용 |
|--------|------|
| `01_data_collection.ipynb` | yfinance + FRED 수집 (완료) |
| `02_feature_engineering.ipynb` | 11개 피처 생성, 타겟 정의, walk-forward 분리 |
| `03_modeling.ipynb` | Baseline + ML/DL 4종 + Best Lookback + Ensemble |
| `04_evaluation.ipynb` | 성능 비교표, ROC Curve, SHAP, 예측 방향 시각화 |

---

## 10. 데이터 수집 도구

| 도구 | 용도 |
|------|------|
| `yfinance` | 주가, 지수, ETF, 금리, 달러 데이터 |
| `fredapi` | 미국 CPI (참고용, 피처에서 제외) |
| `ta` | RSI, MACD 기술적 지표 계산 |
| `shap` | XAI — 피처 기여도 분석 |
