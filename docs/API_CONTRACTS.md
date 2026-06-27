# API CONTRACTS — CoLot-edge (인수인계용)

> 앱 ↔ 백엔드 ↔ 엣지 인터페이스 명세.
> **정직 고지(반드시 먼저 읽을 것)**: 현재 저장소에는 **네트워크 API·메시징·DB가 구현돼 있지 않다.**
> 모든 "인터페이스"는 **같은 프로세스 안의 Python 함수/데이터클래스 호출**이다(인프라는 의도적 비범위 — records/01 D1, records/04).
>
> 따라서 이 문서는 두 부분으로 나눈다:
> - **§1 검증된 in-process 계약** — 소스 실측. 그대로 신뢰 가능.
> - **§2 네트워크 스키마 제안(미구현)** — §1의 데이터클래스에서 1:1 파생한 REST/이벤트 매핑 *제안*.
>   **아직 코드가 없다.** 백엔드 개발자가 구현할 출발점이며, 모든 항목에 `제안(미구현)`을 명시했다.
>
> 표기: 타입은 Python 실측. `분`은 시뮬레이터 t0 기준 분 단위(`Event.start/end`), 절대시각 아님.

---

# §1. 검증된 in-process 계약 (소스 실측)

## 1.1 엣지 파이프라인 진입점 — `kev/pipeline.py`

```python
pipe = build_pipeline(plate_reader, gpu=False)     # adaptive_clf.joblib 로드 + IsolationForest fit
rec  = pipe.on_frame(frame, motion)                # 프레임 1장 처리
rec  = pipe.on_exit(rec, event)                    # 출차 시 이상판정 채움
```

| 함수 | 입력 | 출력 |
|---|---|---|
| `on_frame(frame, motion)` | `frame: np.ndarray (H,W,3) BGR uint8`, `motion: float` | `Record` |
| `on_exit(rec, event)` | `rec: Record`, `event: Event` | `Record`(`anomaly` 필드 채움) |

### `Record` (pipeline.py:21)
| 필드 | 타입 | 의미 |
|---|---|---|
| `env` | `str` | 환경 8종 중 1 (`ENVS`) |
| `mode` | `str` | 센싱 모드 (`rgb_full/rgb_boost/ir/skip`) |
| `run_ocr` | `bool` | 번호판 추론 수행 여부 |
| `power` | `float` | 상대 전력 비용 |
| `plate` | `str \| None` | 인식 번호판(최고 conf), 없으면 None |
| `plate_valid` | `bool` | `PLATE_RE` 통과 여부 |
| `anomaly` | `str \| None` | 출차 후 채워짐 (`normal/unauthorized/fault/anomaly`) |

## 1.2 환경적응 — `kev/adaptive.py`

`AdaptiveSensor.step(bgr, motion=1.0) -> Decision`

### `Decision` (adaptive.py:148)
| 필드 | 타입 | 의미 |
|---|---|---|
| `env` | `str` | 분류된 환경 |
| `mode` | `str` | 센싱 모드 |
| `readable` | `float` | 판독가능성 추정 [0,1] |
| `run_ocr` | `bool` | 추론 게이트 결과 |
| `power` | `float` | `POWER_COST[mode] + (OCR_COST if run_ocr else 0)` |

## 1.3 번호판 인식 — `kev/plate.py`

`PlateReader.read(img) -> list[dict]`. 검출된 박스마다 1개 dict:

| 키 | 타입 | 의미 |
|---|---|---|
| `bbox` | `tuple[int,int,int,int]` | `(x0,y0,x1,y1)` 원본 픽셀 |
| `conf` | `float` | 검출 신뢰도 |
| `text` | `str` | 교정·검증된 번호판 |
| `raw` | `str` | OCR 원문 |
| `valid` | `bool` | `re.match(PLATE_RE, text)` |

보조: `correct_plate(raw) -> (text, valid)`, `char_accuracy(pred, gt) -> float`,
`OnnxYolo.detect(img) -> list[(xyxy: list[4], score: float)]`.
포맷: `PLATE_RE = r"^\d{2,3}[가-힣]\d{4}$"` (예: `12가3456`, `123가4567`).

## 1.4 점유 세션 — `kev/occupancy.py`

### `Event` (occupancy.py:16) — ②/정산/스트리밍 공통 입력
| 필드 | 타입 | 의미 |
|---|---|---|
| `spot` | `int` | 주차면 번호 |
| `start` | `float` | 점유 시작(분) |
| `end` | `float` | 점유 종료(분) |
| `registered` | `bool` | 앱 회원·번호판 등록 차량 여부 |
| `flicker` | `int` | 센서 토글 수(고장 신호) |
| `label` | `str` | 정답 라벨(`normal/unauthorized/fault`) — 평가용 |

## 1.5 이상탐지 — `kev/anomaly.py`

`ParkingAnomalyDetector(cfg).fit(normal_events).predict(events) -> list[Flag]`

### `Flag` (anomaly.py:32)
| 필드 | 타입 | 의미 |
|---|---|---|
| `idx` | `int` | 입력 이벤트 인덱스 |
| `rule` | `str \| None` | 룰 판정(`unauthorized/sensor_fault/None`) |
| `ml_anomaly` | `bool` | IsolationForest 이상 여부 |
| `score` | `float` | 이상 점수(낮을수록 이상) |
| `pred` | `str` | 최종 (`normal/unauthorized/fault/anomaly`) |

## 1.6 실시간 경보 — `kev/streaming.py`

`StreamingMonitor(...).run(events) -> list[Alert]` / `.alert_for(event, idx) -> Alert | None`

### `Alert` (streaming.py:18)
| 필드 | 타입 | 의미 |
|---|---|---|
| `spot` | `int` | 주차면 |
| `time` | `float` | 경보 발생 시각(분) |
| `kind` | `str` | `unauthorized / sensor_fault` |
| `onset` | `float` | 위반 시작 시각 |
| `delay` | `float` | 탐지 지연 = `time - onset` |
| `lead` | `float` | 선행시간 = `end - time`(차량 잔류 여유) |
| `event_idx` | `int` | 원본 이벤트 인덱스 |

## 1.7 정산 — `kev/billing.py`

`settle(event, plate, rate=40, grace=1.0) -> Receipt`

### `Receipt` (billing.py:22)
| 필드 | 타입 | 의미 |
|---|---|---|
| `spot` | `int` | 주차면 |
| `plate` | `str` | 번호판(없으면 `"미인식"`) |
| `minutes` | `int` | 점유 분(올림) |
| `permin_fee` | `int` | 분단위 정산액 |
| `block_fee` | `int` | 30분 블록 환산(경쟁사) |
| `saving` | `int` | 블록 대비 절감 |
| `surcharge` | `int` | 초과 할증(현재 로직상 항상 0) |
| `penalty` | `int` | 무단 과태료 |
| `amount` | `int` | 최종 청구 |
| `status` | `str` | `정산완료 / 무단(미등록·과태료)` |

상수: `RATE=40`(원/분), `BLOCK_MIN=30`, `BLOCK_FEE=1200`, `PENALTY=40000`.
규칙: 등록 = `max(0, dur−grace)`올림 × `rate`; 미등록 = `amount=penalty=PENALTY`.

---

# §2. 네트워크 스키마 제안 (미구현 — 백엔드 구현 출발점)

> 아래는 **§1의 검증된 데이터클래스에서 1:1 파생한 제안**이다. **현재 코드에 존재하지 않는다.**
> 엔드포인트·메서드·전송계층(REST/WebSocket/MQTT)은 전부 `제안(미구현)`. 필드명·타입만 §1과 정합한다.
> 절대시각(`ISO-8601`)·인증·에러 포맷은 백엔드 설계 시 결정할 사항(현재 미정).

## 2.1 차량인식 흐름 — 엣지 → 백엔드  `제안(미구현)`

엣지가 프레임 처리 후(`Record`) 입차/번호판 인식 이벤트를 백엔드로 push 하는 형태를 제안:

```jsonc
// POST /edge/events/recognition   제안(미구현)  ← Record + bbox
{
  "spot": 1,                       // 운영 매핑 필요(현재 Record엔 spot 없음 — Event에만 존재)
  "env": "low_light",              // Record.env
  "mode": "ir",                    // Record.mode
  "run_ocr": true,                 // Record.run_ocr
  "power": 0.55,                   // Record.power
  "plate": "12가3456",             // Record.plate (nullable)
  "plate_valid": true,             // Record.plate_valid
  "ts": "2026-06-28T21:05:00+09:00" // 제안: 절대시각(현재 코드엔 없음)
}
```
> ⚠️ `Record`에는 `spot`·절대시각이 없다(현재 단일 카스토퍼·분단위 시뮬 기준). 멀티-스팟/실시간 운영 시 백엔드가 부여해야 함.

## 2.2 정산 흐름 — 백엔드 → 앱  `제안(미구현)`

출차 후 `settle()`의 `Receipt`를 앱 영수증으로 전달:

```jsonc
// GET /app/receipts/{session_id}   제안(미구현)  ← Receipt
{
  "spot": 1, "plate": "12가3456",
  "minutes": 7,
  "permin_fee": 240, "block_fee": 1200, "saving": 960,
  "surcharge": 0, "penalty": 0,
  "amount": 240,
  "status": "정산완료"
}
```
> 미등록 차량 예: `permin_fee=0, penalty=40000, amount=40000, status="무단(미등록·과태료)"`.

## 2.3 실시간 경보 흐름 — 엣지/백엔드 → 앱  `제안(미구현)`

`StreamingMonitor`의 `Alert`를 앱 push 알림으로:

```jsonc
// WS /app/alerts  또는  POST /app/push   제안(미구현)  ← Alert
{
  "spot": 3,
  "kind": "unauthorized",   // unauthorized | sensor_fault
  "onset_min": 0.0,         // Alert.onset
  "alert_min": 3.0,         // Alert.time
  "delay_min": 3.0,         // Alert.delay (= grace)
  "lead_min": 97.0          // Alert.lead (차량 잔류 여유)
}
```
> 무단 경보 지연 = `unauth_grace`(기본 3분, 번호판 조회 랙). ghost/flicker는 세션종료형이라 실시간 경보 대상 아님(배치 ②가 담당).

## 2.4 매핑 요약 (제안)

| 업무 흐름 | 소스(검증) | 제안 전송 | 비고 |
|---|---|---|---|
| 차량 인식/입차 | `Record` (+`PlateReader.read` dict) | 엣지→백엔드 이벤트 | `spot`·`ts` 백엔드 부여 필요 |
| 이상/무단 판정 | `Flag.pred` | 백엔드 내부/앱 통지 | 출차 후 배치 |
| 실시간 경보 | `Alert` | 엣지/백엔드→앱 push | 주차 '중' |
| 정산 영수증 | `Receipt` | 백엔드→앱 | 분단위 과금 |

> **구현 시 주의(검증된 제약)**: ①`Record`에 `spot`/절대시각 부재 ②번호판 미인식 시 `plate=None`/`"미인식"` 두 표현 혼재(정산은 `"미인식"`) ③시간 단위가 전부 '분'(float) — 운영은 절대시각 변환 필요. 상세 → [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md).
