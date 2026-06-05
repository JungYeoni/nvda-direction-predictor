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

