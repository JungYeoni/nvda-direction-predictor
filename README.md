# nvda-direction-predictor

NVIDIA(NVDA) 내일 종가 방향을 예측하는 이진 분류 프로젝트입니다.

| 항목 | 내용 |
|------|------|
| 예측 종목 | NVIDIA Corporation (NVDA, NASDAQ) |
| 예측 대상 | 내일 종가의 방향 (상승: 1 / 하락·보합: 0) |
| 문제 유형 | 이진 분류 (Binary Classification) |
| 데이터 기간 | 2020.01.02 ~ 2025.05.23 |
| 사용 모형 | Logistic Regression, XGBoost |

---

## 종목 선택 근거

NVIDIA는 **AI 인프라 기업 / 반도체 기업 / 미국 성장주** 세 가지 정체성이 주가에 동시에 작용합니다.  
이 세 가지 축을 반영한 다차원 Feature(15개)로 모형을 설계합니다.

---

## 데이터 분할 전략

```
[    Train    ] [  Validation  ] [ Test ]
 2020.01.02      2023.07.01      2024.07.01   2025.05.23
```

| 구간 | 기간 | 역할 | 거래일(약) |
|------|------|------|-----------|
| Train | 2020.01.02 ~ 2023.06.30 | 모형 학습 | 870일 |
| Validation | 2023.07.01 ~ 2024.06.30 | 하이퍼파라미터 튜닝 | 250일 |
| Test | 2024.07.01 ~ 2025.05.23 | 최종 성능 평가 | 210일 |

> **핵심 스토리**: AI 붐 이전의 패턴으로 학습한 모형이, AI 확산기 이후를 얼마나 정확히 예측할 수 있는가.

---

## Feature 목록 (15개)

| 카테고리 | 변수 | 수집 |
|----------|------|------|
| 기초 데이터 | NVDA_Close, NVDA_Volume, NVDA_Return_1D, NVDA_MA20 | yfinance |
| 기술적 지표 | NVDA_RSI, NVDA_MACD | yfinance 가공 |
| 반도체 섹터 | SOX_Close, TSM_Close | yfinance |
| 거시경제 | US_10Y_Yield, QQQ_Close, DXY_Close, US_CPI | yfinance / FRED API |
| 투자 심리 | VIX_Close | yfinance |
| AI 섹터 | MSFT_Close, META_Close | yfinance |

> CPI는 월간 데이터이므로 발표일 기준으로 다음 발표 전날까지 forward fill 처리합니다.

---

## 환경 설정

Python 3.11 이상, `uv` 기반 의존성 관리를 사용합니다.

```bash
# 가상환경 생성
uv venv nvda-direction-predictor --python 3.11
source nvda-direction-predictor/bin/activate

# 의존성 설치
uv pip install -e ".[dev]"
```

---

## 디렉터리 구조

```text
nvda-direction-predictor/
├── README.md
├── CHANGELOG.md
├── CLAUDE.md
├── pyproject.toml
├── configs/
│   ├── base.yaml          # seed, 분할 기준, 경로 설정
│   ├── dev.yaml
│   └── prod.yaml
├── data/
│   ├── raw/               # 원본 데이터 (git 추적 제외)
│   ├── interim/           # 중간 처리 데이터
│   └── processed/         # 모델 입력용 최종 데이터
├── notebooks/             # EDA, 실험 노트북
├── reports/               # 보고서, 시각화 산출물
├── src/
│   ├── features/          # 데이터 수집 및 피처 생성
│   ├── modeling/          # 모델 학습
│   ├── evaluation/        # 평가 지표
│   └── visualization/     # 시각화
└── tests/
```

---

## 진행 순서

```
1단계  데이터 수집       yfinance(14개) + FRED API(CPI)
2단계  전처리            날짜 병합, RSI/MACD/MA20 생성, 결측치 처리, 타겟 변수 생성
3단계  Train/Val/Test    시간 순서대로 분리 (무작위 분리 금지)
4단계  모형 학습         Logistic Regression (baseline) → XGBoost (튜닝)
5단계  성능 평가         Accuracy, Precision, Recall, F1, Feature Importance
```

---

## 분석 원칙

- `random_state=42` 고정
- 인코더·스케일러는 Train에서만 fit, Val/Test에는 transform만 적용
- rolling/lag 피처는 `shift(1)` 선행 필수 (데이터 누수 방지)
- 원본 데이터 직접 수정 금지 — `df_clean = df.copy()` 패턴 사용

---

## 변경 이력

[`CHANGELOG.md`](CHANGELOG.md) 참고

---

## 라이선스

MIT
