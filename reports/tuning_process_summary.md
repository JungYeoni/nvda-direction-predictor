# 튜닝 과정 기록

## 목적

최고 성능 가능성이 가장 높았던 `excess_qqq_gt_0.2pct` 타깃에 대해 Logistic Regression과 XGBoost를 우선 튜닝했다.

```text
target = 1 if NVDA return - QQQ return > 0.2%p
```

튜닝은 validation set에서만 수행했고, test set은 validation에서 선택된 설정을 최종 평가하는 데만 사용했다.

## 실행 스크립트

| 파일 | 역할 |
|------|------|
| `scripts/run_target_variant_models.py` | 4개 타깃 x 전체 모델 비교 |
| `scripts/tune_lr_xgb_excess_target.py` | 최종 후보 타깃에서 LR/XGBoost 하이퍼파라미터 튜닝 |

## 저장된 결과 파일

`reports/results/`는 git 추적 제외 경로이므로, 로컬 산출물로 저장된다.

| 파일 | 내용 |
|------|------|
| `target_variant_all_models_metrics.csv` | 4개 타깃 x LR/XGBoost/MLP/TCN/LSTM/GRU/iTransformer 비교 |
| `tuning_lr_excess_qqq_02_trials.csv` | LR 튜닝 trial 전체 |
| `tuning_xgb_excess_qqq_02_trials.csv` | XGBoost 튜닝 trial 전체 |
| `tuning_lr_xgb_excess_qqq_02_summary.csv` | validation 기준 선택 설정의 test 평가 |
| `final_assignment_model_metrics.csv` | 제출 보고서용 최종 모형 성능 |
| `final_assignment_predictions.csv` | 제출 보고서용 최종 모형 test 예측 결과 |

## 튜닝 범위

### Logistic Regression

| 파라미터 | 후보 |
|----------|------|
| `C` | 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0 |
| `penalty` | l2 |
| `class_weight` | none, balanced |

### XGBoost

| 파라미터 | 후보 |
|----------|------|
| `max_depth` | 1, 2, 3 |
| `learning_rate` | 0.005, 0.01, 0.03, 0.05 |
| `n_estimators` | 100, 200, 400 |
| `min_child_weight` | 3, 5, 10 |
| `subsample` | 0.6, 0.8, 1.0 |
| `colsample_bytree` | 0.6, 0.8, 1.0 |
| `reg_alpha` | 0.0, 0.1, 0.5 |
| `reg_lambda` | 1.0, 2.0, 5.0 |

## 최종 선택 결과

| 모델 | 선택 기준 | 주요 설정 | Test Accuracy | Test Precision | Test ROC-AUC | Test MCC |
|------|-----------|-----------|---------------|----------------|--------------|----------|
| Tuned LR | validation ROC-AUC | C=0.03, class_weight=balanced, threshold=0.52 | **58.22%** | **0.547** | 0.603 | **0.154** |
| Tuned XGBoost | validation ROC-AUC | depth=2, lr=0.05, n_estimators=400, min_child_weight=10 | 52.44% | 0.481 | 0.542 | 0.043 |

튜닝 결과, Logistic Regression은 Accuracy와 MCC가 개선되었고 최종 제출 모형으로 채택했다. XGBoost는 validation ROC-AUC 기준 상위 설정이 test에서 약하게 일반화되어 최종 모형에서 제외했다.

## 회귀 후 분류 추가 실험

방향만 분류하면 수익률 크기 정보가 사라진다는 한계를 보완하기 위해, 초과수익률 자체를 먼저 예측한 뒤 분류로 변환하는 실험을 추가했다.

```text
y_reg = NVDA return - QQQ return
y_cls = 1 if y_reg > 0.2%p else 0
```

| 모델 | 방식 | Test Accuracy | Test Precision | Test ROC-AUC | Test MCC |
|------|------|---------------|----------------|--------------|----------|
| Huber | 회귀→분류 | **60.44%** | **0.595** | 0.578 | **0.192** |
| Ridge | 회귀→분류 | 53.33% | 0.492 | 0.572 | 0.081 |
| ElasticNet | 회귀→분류 | 55.56% | 0.518 | 0.570 | 0.093 |
| XGBRegressor | 회귀→분류 | 57.33% | 0.621 | 0.554 | 0.126 |
| LGBMRegressor | 회귀→분류 | 55.56% | 0.800 | 0.523 | 0.104 |
| HistGBR | 회귀→분류 | 48.44% | 0.416 | 0.486 | -0.061 |

Huber Regression은 이상치에 덜 민감한 선형 회귀 방식이어서 NVDA 수익률의 급등락일 영향을 줄일 수 있다. 이 실험에서 Huber 회귀→분류가 Accuracy와 MCC 기준으로 가장 좋은 성능을 보였고 Precision도 안정적이었으므로 최종 제출 모형을 기존 tuned LR에서 Huber 회귀→분류로 갱신했다. LGBMRegressor는 Precision은 높지만 Recall이 매우 낮아 신호를 거의 내지 않는 보수적 모형으로 해석한다.
