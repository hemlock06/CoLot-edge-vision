# IMPROVE SUMMARY — CoLot-edge

> 작성: 2026-06-28 야간 무인 개선 작업. 브랜치 **`improve`**(`handoff-prep`에서 분기, push 안 함).
> 범위: [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md) P0/P1 중 **안전·독립검증 가능**한 것만 구현. 설계결정·실데이터 필요분은 [IMPROVE_PROPOSALS.md](IMPROVE_PROPOSALS.md)에 제안만.
> 원칙: 추측 없이 코드 실측·실행 결과 기반. main·handoff-prep 불변(수정/머지/push 없음).

---

## 1. 한 일 (구현 + 검증 잔류)

### P0-3 — 합성 렌더 폰트 이식성 (Windows 하드코딩 → 크로스플랫폼) · 견고화
- **변경**: `kev/plate_synth.py` — 한글 폰트 탐색을 **`KEV_FONT` 환경변수 → `assets/fonts/` 동봉(.ttf/.ttc/.otf) → OS 표준경로(Windows/Linux/macOS)** 순으로 일반화. 폰트 못 찾으면 **탐색 경로·`KEV_FONT` 사용법을 담은 actionable FileNotFoundError**.
- **헬퍼 신설**: `_font_candidates()`(우선순위 목록) · `_resolve_font_path()`(첫 존재 경로) · `_font()`(교체).
- **Windows 기존 동작 불변**: env·동봉 없으면 기존 `_FONT_CANDIDATES`(malgunbd…) 그대로 첫 매칭 → 회귀 없음(테스트로 확인).
- **효과**: 비-Windows(리눅스 CI/도커)에서도 폰트만 설치/지정하면 `render_plate`/`make_scene` 동작 가능. 종전엔 폰트 단계에서 무조건 실패.

### P0-4 — 패키징 부재로 `import kev`가 cwd 의존 → 해소
- **추가**: 루트 **`conftest.py`**(repo 루트를 `sys.path`에 삽입) + **`pyproject.toml`**(`[project] name=kev`, setuptools 패키징, `[tool.pytest.ini_options] testpaths=["tests"]`).
- **효과**: repo 루트가 cwd가 아니어도 pytest가 `kev`를 import. 표준 설치(`pip install .`) 흐름 기반 마련.

### P1-4 — 의존성 버전 핀·재현성 기록
- **추가**: **`requirements.lock.txt`** — 본 작업에서 `pytest 52 passed`로 검증된 .venv의 전체 `pip freeze`(61개 패키지, Python 3.11.15/win32, torch 2.12.1+cpu 등). 헤더에 생성 맥락·용도 기록.
- `requirements.txt`(느슨한 `>=`)는 **건드리지 않음** — 상위호환 설치/‏torch GPU 별도설치 흐름 보존. lock은 "동일결과 재현"용 별도 파일.
- 재현 시드(`SEED=20231016`)는 `kev/config.py`에 이미 중앙화돼 있어 추가 변경 불필요.

### 테스트 추가
- **`tests/test_font_portability.py`**(4개) — 폰트 탐색 우선순위·env 오버라이드·미발견 시 actionable 에러·(폰트 있으면)실렌더를 검증. easyocr/torch 비의존(CI 경량셋 호환).

---

## 2. 테스트 결과 (독립검증)

| 검증 | 결과 |
|---|---|
| 베이스라인 `pytest -q`(개선 전, Windows .venv, 모델 생성됨) | ✅ **48 passed** (43s) |
| 개선 후 `pytest -q`(전체) | ✅ **52 passed** (31s) — 48 기존 + 4 신규, **회귀 0** |
| P0-4 검증: repo 밖 cwd(`D:\`)에서 pytest 수집·import | ✅ **17 passed**(contracts+font) — conftest 없으면 `import kev` 깨질 경로 |
| 적대적 diff 리뷰(새 컨텍스트 서브에이전트 1패스) | ✅ **상/중 결함 없음**, 하 2건 → 둘 다 즉시 반영(Noto `.otf` glob 추가, lock 주석 52 정정) |

- **독립검증 방식**: ①기존 테스트 스위트 통과(잔류 조건) ②테스트 미덮음분(conftest·pyproject·lock)은 적대적 diff 리뷰로 점검. 쿼터/시간 명목 생략 없음.
- 환경: Windows 11, Python 3.11.15, `.venv`. 모델 산출물(`adaptive_clf.joblib`·onnx) 존재 → 모델 의존 테스트도 실제 실행됨(skip 아님).

---

## 3. 미검증 / 한계 (정직 기록)

- **리눅스/맥 실렌더는 본 환경에서 미실행**(플랫폼=win32). 폰트 *탐색 로직*은 테스트로 검증했으나, 실제 리눅스에서 Noto/Nanum로 한글이 깨짐 없이 렌더되는지는 **해당 OS에서 폰트 설치 후 재확인 필요**. (탐색 경로는 배포판 공통값 기준 — 환경에 따라 경로가 다르면 `KEV_FONT`로 지정.)
- **`pip install .`(pyproject 빌드) 실제 설치는 미실행** — pytest import 경로만 검증. 패키징 메타데이터 자체는 표준 형식이나 빌드/배포는 별도 확인 권장.
- `requirements.lock.txt`는 **이 머신의 환경 스냅샷** — 다른 OS/Python 버전에선 일부 휠(특히 torch `+cpu`)이 안 맞을 수 있음(헤더에 명시).

---

## 4. 미구현 (제안만 — [IMPROVE_PROPOSALS.md](IMPROVE_PROPOSALS.md))

설계결정·실데이터·운영정책이 필요해 무인 구현하지 않고 옵션·트레이드오프만 제시:
- **P0-1** 네트워크·서비스 계층(REST/MQTT/DB/인증) — 비구현 범위.
- **P0-2** 대용량 산출물 배포(재생성 vs LFS vs 릴리스 아티팩트).
- **P1-1** INT8 vs FP32 — 타깃 디바이스 선정 의존.
- **P1-2** 날씨 실-전이 — 실 외형 라벨셋 필요.
- **P1-3** 이상탐지 정밀도-재현율 — 운영정책(오탐/누락 비용) 결정.
- **P1-5** API 계약 공백(spot·절대시각·null 표현 통일) — 앱↔백엔드 동시 합의 필요.
- **P2 전체**(하드코딩 상수 환경변수화 등) — 본 야간 범위(P0/P1) 밖, 미착수.

---

## 5. 변경 파일 요약
```
kev/plate_synth.py             (수정) 폰트 탐색 일반화 + actionable 에러
conftest.py                    (신규) sys.path 루트 삽입 → import 이식성
pyproject.toml                 (신규) 패키징 메타 + pytest 설정
requirements.lock.txt          (신규) 검증된 전체 의존성 핀
tests/test_font_portability.py (신규) 폰트 이식성 4 테스트
docs/IMPROVE_SUMMARY.md         (신규) 본 문서
docs/IMPROVE_PROPOSALS.md       (신규) 설계결정 필요분 제안
```
**커밋**: `improve` 브랜치에만. push·머지 없음 — 아침에 hemlo가 diff 확인 후 직접 머지.
