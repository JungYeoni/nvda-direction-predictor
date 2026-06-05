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
y_t = 1  if NVDA_Close_t > NVDA_Close_{t-1}   (오늘 상승)
y_t = 0  if NVDA_Close_t <= NVDA_Close_{t-1}  (오늘 하락/보합)
```

**예측 시점**: t-1일 장 마감 후, t일 시장이 열리기 전  
**사용 정보**: t-1일까지의 모든 시장 데이터 (X_t = shift(1) 적용)  
**예측 대상**: t일 종가가 t-1일 종가보다 높을지 여부

> 모든 피처는 `shift(1)` 적용으로 t-1일 값을 사용한다. t일 데이터는 예측 시점에 알 수 없으므로 피처로 사용하지 않는다.

### 가격 변수 변환 원칙

- `SOX_Close`, `TSM_Close`, `QQQ_Close`, `DXY_Close`, `MSFT_Close`, `META_Close` 등 외부 가격 변수는 모델 입력 전에 수익률 또는 변화율 변수로 변환한다.
- `NVDA_Close` 원값은 타겟 생성과 파생 변수 계산에 사용하고, 최종 모델 입력에는 `NVDA_Return_1D`, `NVDA_MA20`, 이동평균 괴리율 등으로 대체하는 것을 우선한다.
- `US_10Y_Yield`, `VIX_Close`, `US_CPI`는 원값과 변화량 중 Validation 성능 기준으로 선택한다.

### 체크리스트

- [ ] 결측치 확인 (null 비율 출력)
- [ ] 타겟 클래스 비율 확인 (불균형 여부)
- [ ] 이상치 탐지 (거래정지일 등)
- [ ] 모든 피처가 `shift(1)` 이후 계산되었는지 검증
- [ ] 가격 원값 feature가 수익률·변화율·괴리율로 변환되었는지 확인

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

**목표**: 4개 모형을 비교하여 예측력 차이를 분석한다.

| 모형 | 유형 | 역할 |
|------|------|------|
| Logistic Regression | 통계 | 해석 가능한 baseline |
| XGBoost | 트리 앙상블 | 비선형 관계 포착 |
| MLP | 신경망 | 딥러닝 baseline |
| Dilated TCN | 시계열 신경망 | 멀티스케일 시간 패턴 포착 |

### 4-1. Logistic Regression (Baseline)

- 목적: 계수 해석 및 기준 성능 확립
- 정규화: `C` 파라미터 탐색 (L1/L2)
- 평가: Validation 기준 F1-Score

### 4-2. XGBoost

- 목적: 비선형 관계 포착 및 성능 개선
- 튜닝 파라미터: `max_depth`, `learning_rate`, `n_estimators`, `subsample`
- 탐색 방법: `TimeSeriesSplit` 기반 튜닝 (Validation 기준)
- 스케일링: Tree 기반 모형이므로 `StandardScaler` 사용하지 않음
- Feature Importance 시각화 포함

### 4-3. MLP

- 목적: 딥러닝 baseline, 피처 간 비선형 조합 학습
- 구조: FC Layer 3개 + Dropout + BatchNorm
- 입력: 당일 피처 벡터 (tabular)
- 스케일링: `StandardScaler` 적용

### 4-4. Dilated TCN

- 목적: 멀티스케일 시간 패턴 포착
- 구조: Dilated Causal Conv1D (dilation=1,2,4,8) + FC
- 입력: 과거 N일 시퀀스 윈도우
- dilation 적용으로 단기(1~3일) + 중기(1~2주) 패턴 동시 학습
- 스케일링: `StandardScaler` 적용

### 파이프라인 구조

```python
# LR
Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))])

# XGBoost
XGBClassifier(random_state=42, eval_metric="logloss")

# MLP / Dilated TCN
# PyTorch 기반 구현, train/val loop 별도 관리
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

### Naive Baseline

| Baseline | 설명 |
|----------|------|
| Majority Class | Train 구간에서 가장 많은 클래스를 항상 예측 |
| Yesterday Direction | 전일 NVDA 방향과 동일하게 다음 날 방향을 예측 |

> 최종 모델은 위 baseline 대비 성능 개선 여부를 함께 보고한다.

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
| `notebooks/data_collection.ipynb` | yfinance + FRED 수집, raw 저장 |
| `notebooks/eda.ipynb` | 분포, 상관관계, 이상치 탐색 |
| `notebooks/feature_engineering.ipynb` | 파생 변수 생성, 타겟 정의 |
| `notebooks/modeling.ipynb` | 학습, 튜닝, 비교 |
| `notebooks/evaluation.ipynb` | 최종 평가 및 해석 |

---

## 진행 현황

| 단계 | 상태 |
|------|------|
| 1단계 데이터 수집 | 미시작 |
| 2단계 전처리 및 피처 생성 | 미시작 |
| 3단계 데이터 분리 | 미시작 |
| 4단계 모형 학습 | 미시작 |
| 5단계 성능 평가 | 미시작 |
