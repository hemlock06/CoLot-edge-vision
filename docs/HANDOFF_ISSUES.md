# HANDOFF ISSUES — CoLot-edge

> 백엔드·프론트엔드 인수인계 시 알아야 할 이슈·한계·할 일. **2026-06-28 `handoff-prep` 소스 실측 기준.**
> 우선순위: **P0** = 인수인계/실행 차단(먼저 해결) · **P1** = 제품화 전 필수 · **P2** = 개선/스케일업.
> 추측 없이 코드·`records/04_open_issues.md`·실행 결과로 검증된 항목만 기재. 애매한 것은 `⚠️ 확인 필요`.

---

## P0 — 인수인계/실행 차단

### P0-1. 네트워크·서비스 계층 부재 (REST/메시징/DB 미구현)
- **사실**: 저장소 전체가 in-process Python 호출. HTTP 서버·MQTT·DB·인증 코드 없음(의도적 비범위 — records/01 D1, 04 "비범위").
- **영향**: 앱↔백엔드↔엣지 통신은 0부터 구현해야 함. [API_CONTRACTS.md](API_CONTRACTS.md) §2의 제안 스키마가 출발점.
- **할 일(백엔드)**: §2 제안을 실 엔드포인트로 구현. `spot`·절대시각(현재 부재) 부여 정책 결정.

### P0-2. 대용량 산출물 git 미추적 → 클론 직후 데모/일부 테스트 비실행
- **사실**: `.gitignore`에 `data/`, `*.pt`, `*.onnx`, `*.joblib`. 클론하면 `adaptive_clf.joblib`·`best_int8.onnx` 없음.
- **영향**: `python -m kev.cli demo`·`cli plate`는 모델 로드 실패. 모델 의존 테스트 3종은 `pytest.skip`으로 빠짐(에러는 아님).
- **할 일**: `scripts/eval_adaptive.py`(분류기), `scripts/build_plate.py`(검출·INT8) 먼저 실행해 재생성. → [ARCHITECTURE.md](ARCHITECTURE.md) §6.

### P0-3. 합성 렌더가 Windows 폰트 경로 하드코딩 → 비-Windows 실행 불가
- **사실**: `kev/plate_synth.py:20` `_FONT_CANDIDATES`가 전부 `C:\Windows\Fonts\malgunbd.ttf` 등. `_font()`는 후보 없으면 `FileNotFoundError("한글 폰트를 찾지 못함")`.
- **영향**: 비-Windows(리눅스 CI/도커)에서 `render_plate`/`make_scene`/`build_yolo_dataset`/데모/대부분의 `test_kev.py`가 폰트 단계에서 실패. 실측: 본 세션 리눅스류 환경에선 합성 경로 실행 불가, Windows `.venv`에선 35 passed.
- **할 일(소스 수정 필요 — 현 브랜치 범위 밖)**: 폰트 경로를 OS별/환경변수로 추상화하거나 리포지 동봉 폰트 사용. **현재는 수정 금지 정책상 미적용**, 이슈로만 기록.
- **완화(본 브랜치 추가)**: 폰트·중량의존 없는 `tests/test_contracts.py` + Ubuntu CI(`.github/workflows/ci.yml`)로 핵심 로직만 이식성 있게 검증.

### P0-4. 패키징(pyproject/setup) 부재 → import가 cwd 의존
- **사실**: `pyproject.toml`·`setup.py`·`conftest.py` 없음. `import kev`는 repo 루트가 cwd/`sys.path`일 때만 동작(`python -m pytest` 루트 실행으로 성립). 실측: 루트에서 `pytest` 수집 시 `kev` 자체는 import 됨(중량 의존 미설치 환경에선 `cv2` 등에서 멈춤).
- **영향**: 타 디렉터리·표준 패키지 설치 흐름에서 import 깨질 수 있음. CI/도커 작성 시 주의.
- **할 일**: `pyproject.toml`(`[project] name=kev`)로 패키징하거나 루트 `conftest.py` 추가. **소스/구성 수정이라 현 브랜치 미적용**, 이슈로 기록. 본 브랜치 CI는 `python -m pytest` 루트 실행으로 우회.

---

## P1 — 제품화 전 필수

### P1-1. INT8 CPU 지연 역행 (VNNI/NPU 의존)
- **사실**(records/02·04): 정적 INT8 36.5ms > ONNX-FP32 14.1ms (VNNI/NPU 없는 데스크톱). 크기는 3.4MB(7.2×↓).
- **할 일**: 엣지 디바이스 선정 시 "지연 최적=FP32 / 크기 최적=INT8" 트레이드오프 명시. NPU 타깃이면 TFLite/NNAPI·TensorRT 재양자화.

### P1-2. ③ 세부 날씨분류 실-전이 실패 (합성 전용)
- **사실**(records/04, BDD100K 1,053장): **야간/저조도 0.99 실전이**(휘도=물리적) ✅ / 비·눈 등 외형 날씨피처 **실 미전이**(전이 acc 0.36, 실보정 0.52). `streak_coh` 실 주간 도로질감 과발화, `speckle` 실 적설 미발화. glare/backlit/overexposed는 공개 실라벨 부재로 합성 전용.
- **할 일**: 실 배포 전 실 외형정합 라벨셋 필요. **저조도 게이팅 정책 자체는 실 유효** → 우선 활용 가능.

### P1-3. ② 이상탐지 실데이터 미검증 + 정밀도-재현율 트레이드오프
- **사실**: 실 점유×원장 공개셋 부재 → 합성 시뮬 검증에 머묾. 룰+IF 켜면 정밀도 1.0→0.81, 전체 F1은 룰만(0.970)이 더 높음.
- **할 일**: 운영은 "룰=결정, ML=2차 검토 신호" 정책으로. 실 점유/원장 데이터 확보 시 재검증.

### P1-4. 무거운 의존성 + GPU torch 수동 설치 → CI/온보딩 비용
- **사실**: `requirements.txt`에 `torch/ultralytics/easyocr/onnx`(수백 MB~GB). torch GPU 휠은 별도 설치 권장. 전체 `pytest`는 모델 생성 후 약 40초 + 무거운 다운로드.
- **할 일**: 풀 설치는 로컬/Windows 러너. CI는 경량 서브셋(numpy/sklearn/opencv-headless/pillow)으로 계약 테스트만. → 본 브랜치 `ci.yml`이 이 방식.

### P1-5. 계약상 운영 공백 (API 구현 시 결정 필요)
- **사실**(코드 실측): `Record`에 `spot`·절대시각 없음(단일 카스토퍼·분단위 시뮬 전제). 번호판 미인식 표현이 `None`(Record) vs `"미인식"`(Receipt)로 혼재. 시간 단위 전부 '분'(float).
- **할 일(백엔드)**: spot 매핑·절대시각 변환·null 표현 통일 규약 수립. → [API_CONTRACTS.md](API_CONTRACTS.md) §2.4.

---

## P2 — 개선 / 스케일업

- **P2-1. OCR 한글↔숫자 혼동 잔존**: `correct_plate` 혼동맵 + `PLATE_RE` + 다중프레임 투표로 *일부만* 보정(records/04). 실 번호판 다양성에서 추가 오인식 가능.
- **P2-2. 하드코딩 비즈니스 상수**: `RATE=40`·`PENALTY=40000`·`BLOCK_FEE=1200`·`POWER_COST`·각종 임계(`AdaptiveCfg`/`AnomalyCfg`)가 코드 상수. 운영 설정/환경변수화 필요.
- **P2-3. 가정 의존 전력 수치**: 전력 절감 53.9%(민감도 50~56%)는 `POWER_COST` 가정 기반. 헤드라인은 측정값 '추론 호출 −58.5%'로 이미 교체됨. 실 디바이스 소비전류 측정으로 대체 권장.
- **P2-4. 한글 폰트 의존(시각화)**: `kev/plotting.py:use_korean`도 Windows 폰트 경로 사용 → 비-Windows에서 한글 깨짐/폰트 폴백. figs 재생성 시 환경 의존.
- **P2-5. 평가 라벨 결합**: `Event.label`(정답)이 데이터 구조에 포함 — 운영 입력과 평가 입력 분리 시 정리 필요.
- **P2-6. 실 ALPR/악천후 공개셋 확장**(records/04 향후): CCPD 등으로 합성 갭 축소·실세계 OOD 검증.

---

## 검증/회피 요약

| 항목 | 상태 |
|---|---|
| `pytest -q` (Windows `.venv`, 모델 생성됨) | ✅ 35 passed (실측) |
| `tests/test_contracts.py` (본 브랜치 추가, 폰트·중량의존 없음) | ✅ 추가 — Ubuntu CI 대상 |
| `.github/workflows/ci.yml` (본 브랜치 추가) | ✅ 경량 의존 + 계약 테스트 |
| 소스 로직 수정 | ⛔ 없음(정책 준수) — P0-3/P0-4는 이슈 기록만 |
