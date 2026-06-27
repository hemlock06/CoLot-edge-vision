# ARCHITECTURE — CoLot-edge (인수인계용)

> 백엔드·프론트엔드 개발자 인수인계를 위한 아키텍처 문서.
> **모든 내용은 2026-06-28 기준 `handoff-prep` 브랜치 소스 실측**이며, 추측은 배제했다.
> 불확실한 항목은 본문에 `⚠️ 확인 필요`로 표기했다.
> 기존 루트 `ARCHITECTURE.md`·`README.md`·`records/`를 대체하지 않고 보완한다.

---

## 0. 한눈에 — 이 저장소의 정체

코랏(CoLot) 무인 주차관제 **스마트 카스토퍼 1대**의 엣지-비전 처리(입출차 1회 흐름)를,
**공개 모델(YOLOv8n·EasyOCR) + 합성/시뮬레이션 데이터**로 재현한 **레퍼런스 구현**이다.

- 언어/런타임: **Python 3.11**, 단일 패키지 `kev/`. 외부 서비스·DB·메시징 없음(전부 in-process 함수 호출).
- 원 운영 데이터는 비공개 → 합성으로 *의사결정 구조*를 재현·정량 검증하는 것이 목적.
- **검증 상태(실측)**: `pytest -q` → **35 passed** (프로젝트 `.venv`, easyocr/onnx 모델 포함, 약 40초).

> **인수인계 시 가장 먼저 알아야 할 3가지** (상세 → [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md))
> 1. **네트워크/서비스 계층이 없다.** 앱↔백엔드↔엣지 통신은 미구현(의도적 비범위). 인터페이스 = in-process Python 계약뿐. → [API_CONTRACTS.md](API_CONTRACTS.md)
> 2. **대용량 산출물은 git 미추적.** `data/`·`*.pt`·`*.onnx`·`*.joblib`는 `.gitignore`. 클론 직후엔 모델이 없으므로 `scripts/`로 재생성해야 데모·일부 테스트가 돈다.
> 3. **합성 번호판 렌더가 Windows 폰트 경로에 하드코딩**돼 있다(`plate_synth._FONT_CANDIDATES` = `C:\Windows\Fonts\*`). 비-Windows에서 `render_plate`/`make_scene` 계열이 `FileNotFoundError`.

---

## 1. 파이프라인 — env gate(RF) → YOLOv8n INT8 → EasyOCR → anomaly → billing

```
        ┌──────────────────── 카스토퍼 엣지 (kev/pipeline.py) ────────────────────┐
프레임 →│ ③ adaptive   brightness_features(15) → RF 환경분류(8) → policy           │
 +motion│      │                                  → mode(rgb_full/rgb_boost/ir/skip)│
        │      │                                  → run_ocr(bool) · power(float)     │
        │      ▼ run_ocr=True 일 때만 추론                                            │
        │ ① plate     letterbox(320) → YOLOv8n ONNX/INT8(OnnxYolo) → bbox            │
        │      │            → crop → EasyOCR(ko,en) → correct_plate → text·valid      │
        └──────┴──────────────────────┬───────────────────────────────────────────────┘
                                       │  Record(env,mode,run_ocr,power,plate,plate_valid,…)
                  출차 이벤트 Event(점유 세션 start/end + registered 원장)
                                       ▼
        ② anomaly    rule(원장대조+물리위반) ⊕ IsolationForest(센서이상) → pred
                                       ▼
           ├─ streaming  StreamingMonitor : 주차 '중' 실시간 경보 Alert
           └─ billing    settle(Event, plate) : 분단위 정산 Receipt
```

데이터 흐름 코드 경로 (`kev/pipeline.py`):

| 단계 | 호출 | 입력 | 출력 |
|---|---|---|---|
| 프레임 처리 | `CoLotEdgePipeline.on_frame(frame, motion)` | `frame: np.ndarray (H,W,3) BGR uint8`, `motion: float` | `Record` |
| 환경판단 | `AdaptiveSensor.step(frame, motion)` | 〃 | `Decision(env,mode,readable,run_ocr,power)` |
| 번호판 | `PlateReader.read(frame)` | BGR 이미지 | `list[dict{bbox,conf,text,raw,valid}]` (최고 conf 채택) |
| 출차 판정 | `CoLotEdgePipeline.on_exit(rec, event)` | `Record`, `Event` | `Record`(anomaly 채움) |
| 이상탐지 | `ParkingAnomalyDetector.predict([event])` | `list[Event]` | `list[Flag]` |
| 실시간 경보 | `StreamingMonitor.run(events)` | `list[Event]` | `list[Alert]` |
| 정산 | `billing.settle(event, plate)` | `Event`, `plate:str` | `Receipt` |

> `build_pipeline(plate_reader, gpu=False)` (pipeline.py:62) — `data/adaptive_clf.joblib`가 있으면 로드,
> `simulate(n_events=1000)`의 `normal` 세션으로 `IsolationForest`를 fit 해서 파이프라인을 조립한다.
> 분류기 파일이 없으면 `AdaptiveSensor`는 룰(`rule_label`)로 폴백한다.

---

## 2. 모듈 맵 (`kev/`)

| 모듈 | 책임 | 핵심 공개 심볼 (실측) | 무거운 의존 |
|---|---|---|---|
| `config.py` | 상수·경로·설정 dataclass | `ENVS`(8), `MODES`(4), `POWER_COST`, `OCR_COST`, `PLATE_RE`, `PLATE_HANGUL`, `AdaptiveCfg`, `AnomalyCfg`, `PlateCfg`, `SEED=20231016`, `DATA/FIGS/ASSETS` 경로 | 없음 |
| `adaptive.py` ③ | 휘도·악천후 환경적응 | `luminance`, `brightness_features`, `FEATURE_ORDER`(15), `feature_vector`, `rule_label`, `readability`, `Decision`, `AdaptiveSensor` | `cv2` |
| `plate_synth.py` | 합성 한글 번호판·장면·조명·악천후 | `render_plate`, `random_plate_text`, `make_scene`, `relight`, `add_weather`, `apply_env`, `_FONT_CANDIDATES`⚠️ | `cv2`, `PIL` |
| `plate.py` ① | 검출·ONNX·INT8·OCR·포맷교정 | `build_yolo_dataset`, `train_detector`, `export_onnx`, `quantize_static_int8`, `preprocess`, `OnnxYolo`, `PlateReader`, `correct_plate`, `char_accuracy`, `PLATE_ALLOW` | `cv2`, (지연: `ultralytics`/`onnxruntime`/`easyocr`) |
| `tracking.py` ① | 다중프레임 문자투표 | `vote_chars`, `PlateVoter` | 없음(`re`) |
| `occupancy.py` ② | 점유/등록 시뮬레이터 | `Event`(dataclass), `simulate` | `numpy` |
| `anomaly.py` ② | 룰+IsolationForest 이상탐지 | `features`, `FEATS`(5), `Flag`, `ParkingAnomalyDetector` | `sklearn` |
| `streaming.py` ② | 실시간 조기 경보 | `Alert`, `StreamingMonitor` | 없음 |
| `billing.py` | 분단위 정산 | `Receipt`, `settle`, `RATE=40`, `BLOCK_MIN=30`, `BLOCK_FEE=1200`, `PENALTY=40000` | 없음 |
| `pipeline.py` | ③→①→② 통합 | `Record`, `CoLotEdgePipeline`, `build_pipeline` | `joblib` |
| `demo.py` | 통합 데모 시나리오(10종) | `SCENARIO`, `default_plate_reader`, `run_demo` | `cv2`(+matplotlib, 그림 저장 시) |
| `cli.py` | `python -m kev.cli {demo,adaptive,plate}` | `main` | — |
| `plotting.py` | 한글 폰트 matplotlib 설정 | `use_korean` | `matplotlib`, Windows 폰트⚠️ |

**의존 방향(실측)**: `config` ← (모든 모듈). `plate` → `plate_synth`. `tracking` → `plate`.
`anomaly`/`streaming`/`billing`/`pipeline` → `occupancy`. `pipeline` → `adaptive`+`anomaly`+`occupancy`.
순환 의존 없음.

---

## 3. ③ adaptive — 환경 게이트(RF)

- **환경 8종**(`config.ENVS`): 밝기 5 `day_normal·low_light·glare·backlit·overexposed` + 악천후 3 `rain·fog·snow`.
- **피처**(`brightness_features`): 17개 키 반환, 그중 **15개**(`FEATURE_ORDER`)를 RF 입력 벡터로 사용
  (`center`·`border`는 `backlit` 계산용 중간값이라 제외). 휘도 분포(mean/std/p05/p50/p95),
  과포화·암부 비율, 동적범위, 역광, 라플라시안 분산, 글레어 blob, **dark-channel(안개)**,
  HSV 채도, speckle(눈), **streak_coh(각도-불변 빗줄기 응집도)**.
- **분류**(`AdaptiveSensor.classify`): 학습된 분류기(`adaptive_clf.joblib`, sklearn RandomForest)가 있으면 사용,
  없으면 `rule_label`(약지도 룰) 폴백.
- **정책**(`AdaptiveSensor.policy`): `motion < motion_skip(0.012)` → `skip`(절전); `low_light` → `ir`;
  `glare/backlit/overexposed/rain/fog/snow` → `rgb_boost`; 그 외 `rgb_full`.
- **추론 게이팅**(`step`): `run_ocr = mode != "skip" and (mode == "ir" or readable >= readable_min(0.45))`.
- **에너지**: `power = POWER_COST[mode] + (OCR_COST if run_ocr else 0)`.
  `POWER_COST = {rgb_full:1.00, rgb_boost:1.15, ir:0.55, skip:0.05}`, `OCR_COST = 0.80`.

검증 수치(README/records, 합성셋): RF 8클래스 **0.994**(룰 0.472), OOD 2차 생성기 **0.981**, 추론 호출 **−58.5%**(측정), 캡처 유지 100%.

## 4. ① plate — 검출 + OCR + 경량화

- **합성 데이터**: `build_yolo_dataset`(YOLO 라벨·`data.yaml`·`*_gt.csv` 생성), `train_detector`(YOLOv8n, `imgsz=320`).
- **경량화**: `export_onnx`(opset13, simplify) → `quantize_static_int8`(QDQ, per-channel, 캘리브레이션 리더).
  동적 양자화는 conv 미가속으로 폐기됨(records/01 참조).
- **엣지 런타임**(`OnnxYolo`): `onnxruntime` 세션 + YOLOv8 후처리(letterbox 역변환·NMS) 직접 구현 → FP32/INT8 동일 경로.
  출력 `detect(img)` → `list[(xyxy: list[4], score: float)]`.
- **OCR**(`PlateReader.read`): 검출 crop → EasyOCR(`ko,en`, allowlist=`PLATE_ALLOW`) → `correct_plate`(숫자 혼동맵 교정 + `PLATE_RE` 검증).
- **포맷**: `PLATE_RE = r"^\d{2,3}[가-힣]\d{4}$"`. 반환 dict: `{bbox:(x0,y0,x1,y1), conf, text, raw, valid}`.

검증 수치: INT8 **3.4MB(7.2×↓)**, ONNX-CPU FP32 14.1ms / INT8 36.5ms(⚠️ INT8 지연 이득은 VNNI/NPU 의존), OCR char 0.913→0.895.

## 5. ② anomaly + streaming + billing

- **세션**(`occupancy.Event`): `spot:int, start:float(분), end:float, registered:bool, flicker:int, label:str`.
- **피처**(`anomaly.features`, 5차원 `FEATS`): `dur, log_dur, hour, flicker, registered`. (선결제·예약 의존 피처 없음 — 센서 점유 패턴만.)
- **룰**(`ParkingAnomalyDetector.rule`): `dur<min_occupancy_min(2)` → ghost(fault); `flicker>=20` → fault;
  `dur>stuck_min(18h)` → stuck(fault); `not registered` → unauthorized; 그 외 None.
- **ML**: `IsolationForest(contamination=0.03, n_estimators=200)` — 정상 세션 fit, 룰 미검출 중 통계 이상을 `anomaly`로.
- **판정**(`Flag.pred`): `normal / unauthorized / fault / anomaly` 중 하나. 룰 우선, ML은 안전망.
- **실시간**(`StreamingMonitor`): 무단(`unauth_grace=3`분)·stuck만 주차 '중' 조기 경보(ghost/flicker는 세션종료형 → 배치 담당). 출력 `Alert(spot,time,kind,onset,delay,lead,event_idx)`.
- **정산**(`billing.settle`): 등록 차량 = `(점유분 − grace(1분)) × RATE(40원/분)`, 미등록 = `PENALTY(40000원)`.
  `Receipt`에 블록환산(`BLOCK_FEE=1200`/30분)·절감액 포함. 예) 7분 정상 = `(7−1)×40 = 240원`.

검증 수치: 룰만 P=1.000/R=0.941/F1=0.970, 룰+IF P=0.811/R=0.980/F1=0.888(F1은 룰만이 높음 — ML은 재현율 안전망). 스트리밍 주차중 탐지 100%·지연=grace.

---

## 6. 산출물·재현 (클론 직후 필수)

`.gitignore`로 **미추적**(재생성 가능): `data/`, `*.pt`, `*.onnx`, `*.joblib`. `figs/`는 README 참조용으로 추적.

| 산출물 | 생성 스크립트 | 이를 쓰는 코드 |
|---|---|---|
| `data/adaptive_clf.joblib` | `scripts/eval_adaptive.py` | `AdaptiveSensor`/`build_pipeline`/`cli adaptive` |
| `data/runs/plate/weights/best.pt`, `best_int8.onnx` | `scripts/build_plate.py` | `default_plate_reader`/`OnnxYolo`/`cli plate`/`demo` |

미생성 시: `demo`·`cli plate`는 모델 로드 실패, 모델 의존 테스트는 자동 `skip`(코드에 `pytest.skip` 가드 있음).

실행:
```bash
pip install -r requirements.txt        # torch는 GPU 휠 별도 권장
python scripts/eval_adaptive.py        # ③ 분류기·전력 평가 → adaptive_clf.joblib
python scripts/build_plate.py          # ① 학습→ONNX→INT8→벤치 → best*.onnx
python -m kev.cli demo                 # ③→①→② 통합 데모
pytest -q                              # 35 passed (Windows + 모델 생성 후)
```

⚠️ **이식성**: `pytest`/데모는 합성 렌더가 Windows 폰트를 요구한다. 비-Windows에서는 폰트 비의존 계약 테스트
(`tests/test_contracts.py`, 본 브랜치 추가)만 안정 실행된다. → [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md) P0/P1.
