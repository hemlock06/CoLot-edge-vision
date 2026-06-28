"""폰트 이식성(P0-3) 테스트 — 한글 폰트 탐색이 OS·환경변수에 견고한지 검증.

배경: 기존 `_FONT_CANDIDATES`는 전부 Windows 경로 하드코딩이라 비-Windows에서
`render_plate`/`make_scene` 가 폰트 단계에서 FileNotFoundError 로 죽었다(P0-3).
개선: KEV_FONT 환경변수 → assets/fonts 동봉 → OS 표준 경로 순으로 탐색.

의존: numpy, pillow, opencv(-headless). easyocr/torch/onnx 불필요(CI 경량셋 호환).
"""

import pytest

from kev import plate_synth as ps


def test_env_override_is_first_candidate(monkeypatch):
    """KEV_FONT 가 설정되면 후보 목록 맨 앞에 온다(OS 경로보다 우선)."""
    monkeypatch.setenv(ps.FONT_ENV, "/some/custom/font.ttf")
    cands = ps._font_candidates()
    assert cands[0] == "/some/custom/font.ttf"
    # OS 표준 후보도 여전히 뒤에 포함(폴백 유지)
    assert any("Fonts" in c or "fonts" in c for c in cands[1:])


def test_env_override_resolves_existing_file(monkeypatch, tmp_path):
    """KEV_FONT 가 실제 존재 파일을 가리키면 _resolve_font_path 가 그걸 고른다."""
    fake = tmp_path / "myfont.ttf"
    fake.write_bytes(b"\x00\x01")  # 존재만 하면 됨(탐색은 경로 존재 여부만 본다)
    monkeypatch.setenv(ps.FONT_ENV, str(fake))
    assert ps._resolve_font_path() == str(fake)


def test_missing_font_error_is_actionable(monkeypatch):
    """후보가 하나도 없을 때 에러 메시지가 KEV_FONT 사용법을 안내한다."""
    monkeypatch.delenv(ps.FONT_ENV, raising=False)
    monkeypatch.setattr(ps, "_FONT_CANDIDATES", [])
    # 동봉 폰트 디렉터리도 없는 상태로 강제
    monkeypatch.setattr(ps, "_BUNDLED_FONT_DIR", ps.ASSETS / "__no_such_fonts__")
    assert ps._resolve_font_path() is None
    with pytest.raises(FileNotFoundError) as ei:
        ps._font(20)
    assert ps.FONT_ENV in str(ei.value)


def test_render_plate_works_when_font_available():
    """이 플랫폼에 한글 폰트가 있으면 render_plate 가 실제로 동작한다.
    폰트가 없으면(폰트 미설치 CI) 스킵 — 탐색 로직 자체는 위 테스트가 검증."""
    if ps._resolve_font_path() is None:
        pytest.skip("이 환경에 한글 폰트 없음 — 탐색/에러 로직만 검증(위 테스트)")
    img = ps.render_plate("12가3456")
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0
