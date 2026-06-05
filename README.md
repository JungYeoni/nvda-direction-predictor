# nvda-direction-predictor

NVIDIA(NVDA)의 다음 거래일 종가 방향을 예측하는 이진 분류 프로젝트입니다.

| 항목 | 내용 |
|------|------|
| 예측 종목 | NVIDIA Corporation (NVDA, NASDAQ) |
| 예측 대상 | 다음 거래일 종가 방향 (상승: 1 / 하락·보합: 0) |
| 문제 유형 | 이진 분류 (Binary Classification) |
| 데이터 기간 | 2020.01.09 ~ 2025.05.22 |
| 사용 모형 | Logistic Regression, XGBoost, MLP, Dilated TCN |

---

## 종목 선택 근거

NVIDIA는 **AI 인프라 기업 / 반도체 기업 / 미국 성장주** 세 가지 정체성이 주가에 동시에 작용합니다.
이 세 가지 축을 반영한 다차원 피처(22개)로 모형을 설계합니다.

---

## 데이터 분할 전략

```
[    Train    ] [  Validation  ] [ Test ]
 2020.01.09      2023.07.01      2024.07.01   2025.05.22
```

| 구간 | 기간 | 거래일 |
|------|------|--------|
| Train | 2020.01.09 ~ 2023.06.30 | 875일 |
| Validation | 2023.07.01 ~ 2024.06.30 | 250일 |
| Test | 2024.07.01 ~ 2025.05.22 | 225일 |

> **핵심 스토리**: AI 붐 이전의 패턴으로 학습한 모형이, AI 확산기 이후를 얼마나 정확히 예측할 수 있는가.

---

## 피처 구성 (22개)

| 범주 | 피처 |
|------|------|
| NVDA 기술지표 | Return_1D, MA20_ratio, Volume_ratio, RSI, MACD |
| 상대강도 | NVDA_vs_QQQ, NVDA_vs_SOX |
| 변동성 | NVDA_RealVol_20d, VIX, VIX_delta |
| 섹터/종목 | SOX_Return, TSM_Return, QQQ_Return, MSFT_Return, META_Return |
| 매크로 | US_10Y_Yield, DXY_Return, US_CPI |
| 이벤트 캘린더 | is_nvda_post_earnings, is_nvda_earnings_eve, is_fomc_day, is_cpi_day |

시장 데이터는 lookahead 방지를 위해 `shift(1)` 적용, 캘린더 피처는 사전에 알려진 일정이므로 shift 없음. 즉 `X_t`는 전일 장마감까지의 정보이고, `y_t`는 당일 종가가 전일 종가보다 상승했는지 여부입니다.

---

## 최종 성능 해석 (Test Set, 2024.07 ~ 2025.05)

| 모델 | Accuracy | ROC-AUC | F1 |
|------|----------|---------|----|
| **Logistic Regression** | 52.44% | **0.558** | 0.686 |
| XGBoost | 52.44% | 0.506 | 0.658 |
| MLP | 52.44% | 0.536 | 0.688 |
| Dilated TCN | 53.17% | 0.466 | 0.694 |
| Majority baseline | 52.4% | — | — |

최종 해석은 ROC-AUC 기준으로 가장 안정적인 **Logistic Regression**을 중심으로 한다. Dilated TCN은 F1과 Accuracy가 높지만 `Recall=1.0`, `ROC-AUC=0.466`으로 상승 예측에 치우친 퇴화 신호가 있어 최종 우위 모델로 보지 않는다.

고확신 필터는 validation 기준으로 threshold를 선택하면 XGBoost `threshold=0.49`, test precision 52.8%, coverage 86.7%가 나온다. 기존 `threshold=0.52`의 test precision 56.3%는 test 기준 사후 선택 성격이 있어 최종 성능으로 사용하지 않는다.

---

## 환경 설정

Python 3.11 이상, `uv` 기반 의존성 관리를 사용합니다.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 디렉터리 구조

```
nvda-direction-predictor/
├── data/
│   ├── raw/               # 원본 데이터
│   └── processed/         # 모델 입력용 피처 데이터
├── models/                # 학습된 모델 파일
├── notebooks/
│   ├── data_collection.ipynb
│   ├── eda.ipynb
│   ├── modeling.ipynb
│   ├── evaluation.ipynb
│   ├── ablation.ipynb
│   └── ensemble.ipynb
├── reports/
│   ├── eda_summary.md
│   ├── modeling_summary.md
│   ├── experiment_summary.md
│   ├── figures/
│   └── results/
└── src/
    ├── features/
    ├── modeling/
    ├── evaluation/
    └── visualization/
```

---

## 분석 원칙

- `random_state=42` 고정
- 인코더·스케일러는 Train에서만 fit, Val/Test에는 transform만 적용
- rolling/lag 피처는 `shift(1)` 선행 (데이터 누수 방지)
- 원본 데이터 직접 수정 금지 — `df_clean = df.copy()` 패턴

---

## 변경 이력

[`CHANGELOG.md`](CHANGELOG.md) 참고

---

## 라이선스

MIT
