# ARCHITECTURE — CoLot-edge

스마트 카스토퍼 1대의 입출차 처리를 세 모듈로 재현한다. 각 모듈은 독립
실행·평가 가능하며 `pipeline.py`에서 한 흐름으로 결합된다.

## 데이터 흐름
```
        ┌─────────────── 카스토퍼 엣지 ───────────────┐
프레임 →│ ③ adaptive  휘도피처 → 환경분류(RF) → 정책   │→ mode(RGB/IR/skip), run_ocr
        │     │ run_ocr=True                          │
        │ ① plate    letterbox → YOLOv8n(ONNX/INT8)   │→ bbox → crop → EasyOCR → 번호판
        └──────────────────┬───────────────────────────┘
                           │ 점유 세션(start,end) + 앱 원장(예약/결제)
        ② anomaly  룰(원장대조) ⊕ IsolationForest(센서이상) → 무단/초과/고장
```

## ③ adaptive — 휘도·악천후 환경적응
- **환경 8종**: 밝기 5(`day_normal·low_light·glare·backlit·overexposed`)
  + 악천후 3(`rain·fog·snow`). 악천후는 원 휘도 특허 범위를 넘는
  실외 로버스트니스 확장으로 명시.
- **피처**(`brightness_features`, 13차원): 휘도(Y) 분포(mean/std/p05/p50/p95),
  과포화·암부 비율, 동적범위, 역광(중앙-주변), 라플라시안 분산(국소대비),
  글레어 연결요소 + **dark-channel prior(안개 지표)·HSV 채도(탈색)**.
- **악천후 합성**(`add_weather`): 비(빗줄기 모션블러·렌즈 물방울·탈색),
  안개(대기산란 모델 `img·t + A·(1−t)`), 눈(밝은 송이 + 블러).
- **분류**: RandomForest(8클래스). 약지도 라벨러(`rule_label`)는 폴백·베이스라인
  (안개는 dark-channel로 잡지만 비/눈은 약함 → RF가 보강). 평가는
  **장면 단위 분리**로 train/test 누수 차단.
- **정책**(`policy`): 모션<임계 → `skip`(듀티사이클 절전); 저조도 → `ir`;
  글레어/역광/과노출 → `rgb_boost`; 그 외 `rgb_full`. 추론(검출+OCR)은
  센싱 ON & 판독가능(또는 IR)일 때만 → 낭비 추론 제거.
- **에너지 모델**: `POWER_COST[mode] + OCR_COST·run_ocr`. 상시 RGB 대비
  절감률·캡처 유지율로 평가.

## ① plate — 검출 + OCR + 경량화
- **합성 데이터**(`plate_synth`): 맑은 고딕 한글 번호판(유효 포맷) → 노면 배경에
  원근 워프 합성 + 조명 증강. YOLO 라벨(`build_yolo_dataset`) 자동 생성.
- **검출**: YOLOv8n, imgsz=320(엣지). `export_onnx`(opset13) →
  `quantize_static_int8`(QDQ, per-channel, 50장 캘리브레이션).
- **엣지 런타임**(`OnnxYolo`): onnxruntime 세션 + YOLOv8 후처리(디코드·NMS)를
  직접 구현 → FP32/INT8 동일 경로로 공정 비교.
- **OCR**(`PlateReader`): EasyOCR(ko,en) + 한국 번호판 정규식 검증.
- **경량화 관점**: 크기(7.2×↓)·지연·검출/문자정확도를 백엔드별 실측.
  INT8 지연 이득은 VNNI/NPU 하드웨어 의존 — 측정으로 명시.

## ② anomaly — 불법주차 이상탐지
- **시뮬**(`occupancy.simulate`): normal/unauthorized/overstay/fault 혼합.
  fault는 명백형(stuck≥20h·flicker≥22·ghost<2분)과 **subtle형**(원장 정상 +
  약한 센서이상: flicker 8–18, ghost 2.5–4.5분)으로 나눠 ML 가치를 검증.
- **룰**: 점유 세션을 예약/결제 원장과 대조 → 무단(미예약)·초과(결제초과+grace)
  결정론적 판정 + 물리위반(초단/장시간/flicker) 고장.
- **ML**: 정상 세션으로 IsolationForest 적합 → 룰이 못 잡는 subtle 고장을
  통계 이상으로 포착. 룰 우선, 미검출분만 ML 보강(2단 방어).
- **평가**: 이진 P/R/F1 + 유형별 재현율 + 혼동행렬. 정밀도-재현율 트레이드오프
  (ML이 고장 재현율↑, 정밀도 소폭↓)를 정직하게 노출.

## 설계 원칙
- **정직성**: 비공개 데이터를 지어내지 않음 — 합성으로 구조 재현, 한계 명시.
- **방어가능성**: 거대 인프라(MQTT/DB/CI) 없이 핵심 AI 3개에 집중,
  각 결정과 수치를 설명 가능한 규모로 유지.
- **재현성**: 고정 시드, 장면 단위 분할, 산출물(joblib/onnx/json) 캐시.
