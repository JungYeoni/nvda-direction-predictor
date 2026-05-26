# NVDA Direction Predictor — 프로젝트 단계별 계획

> 참고 문서: `nvda_project_overview.md`  
> 데이터 기간: 2020.01.02 ~ 2025.05.23  
> 모형: Logistic Regression (baseline), XGBoost

---

## 전체 흐름

```
1단계  데이터 수집
   ↓
2단계  전처리 및 피처 생성
   ↓
3단계  Train / Validation / Test 분리
   ↓
4단계  모형 학습 (Logistic Regression → XGBoost)
   ↓
5단계  성능 평가 및 해석
```

---

## 1단계 — 데이터 수집

**목표**: 15개 Feature를 날짜 기준으로 정합성 있게 수집한다.

### 수집 대상

| 소스 | 수집 변수 | 도구 |
|------|-----------|------|
| yfinance | NVDA, SOX, TSM, QQQ, DXY, VIX, MSFT, META, US_10Y_Yield | `yfinance` |
| FRED API | US_CPI (월간, CPIAUCSL) | `fredapi` |

### 처리 기준

- 수집 기간: `2019.12.01 ~` (lag 피처 생성 여유분 포함)
- 날짜 기준: 미국 주식 거래일 기준으로 정렬, 주말·공휴일 제거
- CPI: 월 1회 발표 → 발표일 값을 다음 발표 전날까지 **forward fill**

### 산출물

- `data/raw/prices_raw.csv` — yfinance 원본
- `data/raw/cpi_raw.csv` — FRED 원본

---

## 2단계 — 전처리 및 피처 생성

**목표**: 모델 입력 피처 15개를 완성하고 타겟 변수를 생성한다.

### 파생 변수 생성

| 변수 | 계산식 | 주의 |
|------|--------|------|
| `NVDA_Return_1D` | `(Close_t - Close_{t-1}) / Close_{t-1}` | shift(1) 필수 |
| `NVDA_MA20` | 20일 단순이동평균 | shift(1) 이후 rolling |
| `NVDA_RSI` | RSI(14), `ta` 라이브러리 사용 | shift(1) 필수 |
| `NVDA_MACD` | EMA(12) - EMA(26), `ta` 라이브러리 사용 | shift(1) 필수 |

### 타겟 변수 정의

```
y_t = 1  if NVDA_Close_{t+1} > NVDA_Close_t
y_t = 0  if NVDA_Close_{t+1} <= NVDA_Close_t
```

> `y_t`는 **내일 종가 방향**이므로 피처는 모두 당일 또는 그 이전 값만 사용해야 한다.

### 체크리스트

- [ ] 결측치 확인 (null 비율 출력)
- [ ] 타겟 클래스 비율 확인 (불균형 여부)
- [ ] 이상치 탐지 (거래정지일 등)
- [ ] 모든 피처가 `shift(1)` 이후 계산되었는지 검증

### 산출물

- `data/processed/features.csv` — 피처 + 타겟 통합 데이터셋

---

## 3단계 — Train / Validation / Test 분리

**목표**: 시간 순서를 유지한 분리로 데이터 누수를 원천 차단한다.

### 분할 기준

| 구간 | 기간 | 거래일(약) |
|------|------|-----------|
| Train | 2020.01.02 ~ 2023.06.30 | 870일 |
| Validation | 2023.07.01 ~ 2024.06.30 | 250일 |
| Test | 2024.07.01 ~ 2025.05.23 | 210일 |

### 전처리 적용 원칙

- 스케일러(`StandardScaler`)는 **Train에서만 fit**, Val/Test에는 transform만 적용
- 인코더 동일 원칙 적용
- `random_state=42` 고정 (분할이 필요한 경우)

### 산출물

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

---

## 4단계 — 모형 학습

**목표**: Logistic Regression을 baseline으로 먼저 확립하고, XGBoost로 성능을 개선한다.

### 4-1. Logistic Regression (Baseline)

- 목적: 계수 해석 및 기준 성능 확립
- 정규화: `C` 파라미터 탐색 (L1/L2)
- 평가: Validation 기준 F1-Score

### 4-2. XGBoost

- 목적: 비선형 관계 포착 및 성능 개선
- 튜닝 파라미터: `max_depth`, `learning_rate`, `n_estimators`, `subsample`
- 탐색 방법: `optuna` 또는 `GridSearchCV` (Validation 기준)
- Feature Importance 시각화 포함

### 파이프라인 구조

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42)),
])
```

### 산출물

- `src/modeling/train_lr.py`
- `src/modeling/train_xgb.py`
- `reports/model_comparison.md`

---

## 5단계 — 성능 평가 및 해석

**목표**: Test 데이터 기준 최종 성능을 측정하고, 모형의 한계를 명시한다.

### 평가 지표

| 지표 | 선택 이유 |
|------|-----------|
| Accuracy | 전체 예측 정확도 |
| Precision | 상승 예측의 정밀도 (거짓 신호 최소화) |
| Recall | 실제 상승을 얼마나 포착했는가 |
| F1-Score | Precision-Recall 균형 (주요 기준) |
| ROC-AUC | 임계값 무관 분리 성능 |

### 추가 분석

- Confusion Matrix 시각화
- Feature Importance (XGBoost)
- Logistic Regression 계수 해석
- 시기별 예측 정확도 추이 (AI 붐 전후 비교)

### 한계점 명시 항목

- 과거 패턴 기반 예측의 구조 변화 취약성
- 거시경제 이벤트(금리 발표, 실적 쇼크) 미반영
- 거래 비용·슬리피지 미고려

### 산출물

- `reports/evaluation_report.md`
- `reports/figures/` — confusion matrix, feature importance, ROC curve

---

## 노트북 구성 계획

| 파일 | 내용 |
|------|------|
| `notebooks/01_data_collection.ipynb` | yfinance + FRED 수집, raw 저장 |
| `notebooks/02_eda.ipynb` | 분포, 상관관계, 이상치 탐색 |
| `notebooks/03_feature_engineering.ipynb` | 파생 변수 생성, 타겟 정의 |
| `notebooks/04_modeling.ipynb` | 학습, 튜닝, 비교 |
| `notebooks/05_evaluation.ipynb` | 최종 평가 및 해석 |

---

## 진행 현황

| 단계 | 상태 |
|------|------|
| 1단계 데이터 수집 | 미시작 |
| 2단계 전처리 및 피처 생성 | 미시작 |
| 3단계 데이터 분리 | 미시작 |
| 4단계 모형 학습 | 미시작 |
| 5단계 성능 평가 | 미시작 |
