# 02 — 평가 결과

모든 수치는 고정 시드(`SEED=20231016`) 합성 평가셋 기준. 산출 JSON: `data/*_metrics.json`,
`data/plate_bench.json`, `data/weather_ocr.json`.

## ③ 휘도·악천후 환경적응

### (A) 환경분류 (8클래스, 장면단위 분할)
| 분류기 | accuracy | macro-F1 |
|---|---|---|
| RandomForest | **0.972** | 0.972 |
| 손룰 베이스라인 | 0.472 | 0.366 |

(밝기 5종만일 때는 RF 0.995 / 룰 0.640 — 악천후 추가로 난이도 상승.)
도식: `figs/adaptive_confusion.png`, `figs/adaptive_decisions.png`

### (B) 전력-커버리지 (상시 RGB 대비)
| 지표 | 값 |
|---|---|
| 추론 전력 절감 | **−53.9%** |
| 낭비 OCR 호출 감소 | −58.5% |
| 회수가능 캡처 유지율 | **100%** |

도식: `figs/adaptive_tradeoff.png`

### (C) 악천후 OCR 열화 (정책 근거)
| 조건 | 완전일치 | 문자정확도 |
|---|---|---|
| 맑음 | 82.5% | 97.0% |
| 비 | 70.0% | 95.8% |
| 안개 | 82.5% | 96.7% |
| 눈 | **62.5%** | 93.6% |

도식: `figs/weather_ocr.png`

## ① 번호판 검출 + OCR + 경량화

| 백엔드 | 크기 | 지연(1장) | 검출율(IoU≥.5) |
|---|---|---|---|
| PyTorch GPU (FP32) | 24.4 MB | 14.5 ms | 1.00 |
| PyTorch CPU (FP32) | 24.4 MB | 40.1 ms | — |
| ONNX CPU (FP32) | 12.1 MB | **14.1 ms** | 1.00 |
| ONNX CPU INT8 (엣지) | **3.4 MB** | 36.5 ms | 0.96 |

OCR 종단(EasyOCR ko): 클라우드 exact 0.73 / char 0.913, 엣지(INT8) exact 0.75 / char 0.895.
학습 검출기 mAP50≈0.99(320px, 합성). 도식: `figs/plate_quant.png`, `figs/plate_samples.png`

## ② 불법주차 이상탐지 (test=800, anomalies≈209)

| 구성 | Precision | Recall | F1 |
|---|---|---|---|
| 룰만 | 1.000 | 0.842 | 0.914 |
| 룰 + IsolationForest | 0.866 | 0.895 | 0.880 |

유형별 재현율(룰→룰+ML): 무단점유 100→100 · 초과주차 74.0→77.9 · **센서고장 70.5→88.6**.
도식: `figs/anomaly_recall.png`, `figs/anomaly_confusion.png`, `figs/anomaly_sessions.png`
