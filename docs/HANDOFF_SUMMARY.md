# HANDOFF SUMMARY — CoLot-edge

> 백엔드·프론트엔드 개발자 인수인계 **시작 문서**. 5분 안에 전체를 파악하고 다음 문서로 분기.
> 작성: 2026-06-28, `handoff-prep` 브랜치. **모든 수치·구조는 소스 실측**(추측 배제, 애매한 건 `⚠️`).

## 1. 한 줄 정의
코랏 무인 주차관제 **스마트 카스토퍼**의 엣지-비전 파이프라인
(**환경게이트(RF) → YOLOv8n INT8 → EasyOCR → 이상탐지 → 정산**)을, 공개 모델 + 합성/시뮬 데이터로 재현한 **Python 레퍼런스 구현**.

## 2. 문서 인덱스 (읽는 순서)
| 순서 | 문서 | 누가 |
|---|---|---|
| 1 | **HANDOFF_SUMMARY.md** (이 문서) | 전원 |
| 2 | [ARCHITECTURE.md](ARCHITECTURE.md) — 파이프라인·모듈맵·데이터흐름 | 전원 |
| 3 | [API_CONTRACTS.md](API_CONTRACTS.md) — in-process 계약(검증) + 네트워크 스키마 제안(미구현) | 백엔드·프론트 |
| 4 | [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md) — P0/P1/P2 이슈·한계·할 일 | 전원 |
| 보조 | 루트 `README.md`, `decisions.md`, `records/01~04` | 배경·결정 근거 |

## 3. 저장소 구조 (실측)
```
kev/            핵심 패키지 — adaptive③ · plate① · anomaly② · streaming · billing · pipeline · cli · demo
scripts/        평가·빌드 스크립트(13개) — eval_adaptive(분류기 생성) · build_plate(검출·INT8 생성) 등
tests/          test_kev.py(35, 기존) + test_contracts.py(13, 본 브랜치 추가·폰트 비의존)
.github/workflows/ci.yml   본 브랜치 추가 — Ubuntu 경량 계약 테스트
docs/           본 인수인계 문서 묶음(본 브랜치 추가)
records/        설계결정·평가·실행·이슈 기록
requirements.txt  torch/ultralytics/easyocr/onnx/opencv/sklearn …
data/·*.pt·*.onnx·*.joblib   .gitignore(미추적) — scripts/로 재생성
```

## 4. 빠른 시작
```bash
# (Windows 권장 — 합성 렌더가 Windows 폰트 의존, HANDOFF_ISSUES P0-3)
pip install -r requirements.txt
python scripts/eval_adaptive.py     # data/adaptive_clf.joblib 생성
python scripts/build_plate.py       # data/runs/plate/weights/best*.onnx 생성
python -m kev.cli demo              # ③→①→② 통합 데모
pytest -q                          # 35 passed (모델 생성 후)

# 어디서든(폰트·중량모델 없이) 핵심 로직만 빠르게:
pip install numpy scikit-learn opencv-python-headless pillow joblib pytest
python -m pytest tests/test_contracts.py -q   # 13 passed
```

## 5. 검증 상태 (이 인수인계 작업에서 실측)
- `pytest -q` (Windows `.venv`, 모델 생성됨) → **35 passed** ✅
- `python -m pytest tests/test_contracts.py` → **13 passed** ✅ (본 브랜치 추가, 폰트·중량의존 0)
- 소스 로직 수정 **없음**(추가만). main 커밋·push **없음**.

## 6. 인수인계자가 가장 먼저 알아야 할 3가지 (상세 → HANDOFF_ISSUES)
1. **네트워크 계층이 없다(P0-1).** 앱↔백엔드↔엣지 통신 미구현 = 의도적 비범위. 인터페이스는 in-process Python 계약뿐. 백엔드는 [API_CONTRACTS](API_CONTRACTS.md) §2 제안 스키마부터 구현.
2. **모델/데이터가 git에 없다(P0-2).** 클론 직후 `scripts/`로 재생성해야 데모·일부 테스트가 돈다.
3. **합성 렌더가 Windows 폰트 하드코딩(P0-3).** 비-Windows에선 `make_scene`/`render_plate` 계열이 실패 → CI는 폰트 비의존 계약 테스트만 실행.

## 7. 역할별 진입점
- **백엔드**: API_CONTRACTS §1(검증된 dataclass) → §2(REST/이벤트 제안 구현). HANDOFF_ISSUES P0-1, P1-5(spot·절대시각·null 표현).
- **프론트엔드**: API_CONTRACTS §2.2 정산 영수증(`Receipt`)·§2.3 실시간 경보(`Alert`) 스키마. 차량인식 결과(`Record`)는 §2.1.
- **ML/엣지**: ARCHITECTURE §3~5 + HANDOFF_ISSUES P1-1(INT8 지연), P1-2(날씨 실전이 한계).

## 8. 본 브랜치(`handoff-prep`) 추가물 (추가만 — 기존 소스 불변)
- `docs/ARCHITECTURE.md`, `docs/API_CONTRACTS.md`, `docs/HANDOFF_ISSUES.md`, `docs/HANDOFF_SUMMARY.md`
- `tests/test_contracts.py` (폰트·중량모델 비의존 계약 테스트 13개)
- `.github/workflows/ci.yml` (Ubuntu 경량 + Windows 전체(수동))
