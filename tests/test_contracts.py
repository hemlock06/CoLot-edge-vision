"""인수인계용 계약/불변식 테스트 — 폰트·중량모델 비의존(이식성).

목적: docs/API_CONTRACTS.md 의 데이터 계약과 docs/ARCHITECTURE.md 의 파이프라인
불변식을 코드로 고정한다. 기존 tests/test_kev.py 를 수정하지 않고 보완한다.

설계 제약(왜 이 테스트만 CI에서 도는가):
  - kev/plate_synth 의 render_plate/make_scene 는 Windows 폰트 경로에 의존(P0-3)
    → 합성 이미지 렌더를 호출하지 않는다(numpy 난수 BGR 배열만 사용).
  - easyocr/onnxruntime/ultralytics/torch 미설치 환경에서도 통과한다(import 안 함).
의존: numpy, scikit-learn, opencv-python(-headless), pillow. → .github/workflows/ci.yml.
"""

import re
import numpy as np
import pytest

from kev.config import (
    ENVS,
    LIGHT_ENVS,
    WEATHER_ENVS,
    MODES,
    POWER_COST,
    OCR_COST,
    PLATE_RE,
    PLATE_HANGUL,
    SEED,
    AdaptiveCfg,
    AnomalyCfg,
)


def _rand_bgr(h=120, w=240, seed=0):
    """폰트 비의존 합성 프레임(난수 BGR uint8)."""
    g = np.random.default_rng(seed)
    return g.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---- config 불변식 (문서 표와 정합) ----
def test_env_and_mode_cardinality():
    assert len(ENVS) == 8 and ENVS == LIGHT_ENVS + WEATHER_ENVS
    assert len(LIGHT_ENVS) == 5 and len(WEATHER_ENVS) == 3
    assert MODES == ["rgb_full", "rgb_boost", "ir", "skip"]


def test_power_cost_covers_all_modes():
    assert set(POWER_COST) == set(MODES)
    assert POWER_COST["skip"] < POWER_COST["ir"] < POWER_COST["rgb_full"]
    assert isinstance(OCR_COST, float) and OCR_COST > 0


def test_plate_regex_contract():
    for ok in ("12가3456", "123가4567"):
        assert re.match(PLATE_RE, ok)
    for bad in ("1가3456", "12345678", "12가345", "AB가3456"):
        assert not re.match(PLATE_RE, bad)
    assert len(PLATE_HANGUL) >= 1
    assert SEED == 20231016


# ---- ③ adaptive (폰트 비의존) ----
def test_feature_order_is_15_and_vector_matches():
    from kev.adaptive import brightness_features, FEATURE_ORDER, feature_vector

    assert len(FEATURE_ORDER) == 15
    feats = brightness_features(_rand_bgr())
    assert set(FEATURE_ORDER).issubset(feats)
    assert feature_vector(feats).shape == (15,)


def test_readability_in_unit_range():
    from kev.adaptive import brightness_features, readability

    r = readability(brightness_features(_rand_bgr(seed=1)))
    assert 0.0 <= r <= 1.0


def test_decision_contract_and_power_formula():
    from kev.adaptive import AdaptiveSensor, Decision

    # 모션 0 → skip → run_ocr False → power == POWER_COST['skip']
    d = AdaptiveSensor().step(_rand_bgr(seed=2), motion=0.0)
    assert isinstance(d, Decision)
    for f in ("env", "mode", "readable", "run_ocr", "power"):
        assert hasattr(d, f)
    assert d.mode == "skip" and d.run_ocr is False
    assert d.power == pytest.approx(POWER_COST["skip"])
    # 일반 프레임: power = POWER_COST[mode] + (OCR_COST if run_ocr else 0)
    d2 = AdaptiveSensor().step(_rand_bgr(seed=3), motion=0.5)
    expected = POWER_COST[d2.mode] + (OCR_COST if d2.run_ocr else 0.0)
    assert d2.power == pytest.approx(expected)


# ---- ① plate: 포맷 교정(폰트·OCR 비의존) ----
def test_correct_plate_and_allowlist():
    from kev.plate import correct_plate, char_accuracy, PLATE_ALLOW

    assert correct_plate("12가3456") == ("12가3456", True)
    assert correct_plate("1 2가 3456") == ("12가3456", True)  # 공백 제거
    assert correct_plate("12O3456")[1] is False  # 한글 없음 → invalid
    assert correct_plate("12가34I6")[0] == "12가3416"  # I→1 교정
    assert char_accuracy("12가3456", "12가3456") == 1.0
    assert char_accuracy("", "12가3456") == 0.0
    assert set("0123456789").issubset(set(PLATE_ALLOW))


def test_preprocess_shape():
    from kev.plate import preprocess

    x, r, dx, dy = preprocess(np.zeros((480, 640, 3), np.uint8), 320)
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
    assert 0.0 < r <= 1.0


# ---- ② occupancy / anomaly ----
def test_event_and_simulate_contract():
    from kev.occupancy import simulate, Event

    evs = simulate(n_events=300, seed=1)
    assert len(evs) == 300
    assert {"normal", "unauthorized", "fault"}.issubset({e.label for e in evs})
    e = evs[0]
    for f in ("spot", "start", "end", "registered", "flicker", "label"):
        assert hasattr(e, f)
    assert isinstance(e, Event) and e.end >= e.start


def test_anomaly_rule_and_flag_contract():
    from kev.anomaly import ParkingAnomalyDetector, features, FEATS, Flag
    from kev.occupancy import Event, simulate

    det = ParkingAnomalyDetector()
    assert det.rule(Event(1, 0, 40, False, 0, "x")) == "unauthorized"
    assert det.rule(Event(1, 0, 30, True, 40, "x")) == "sensor_fault"  # flicker
    assert det.rule(Event(1, 0, 40, True, 0, "x")) is None  # 정상
    assert len(features(Event(1, 0, 40, True, 0, "x"))) == len(FEATS) == 5

    evs = simulate(n_events=400, seed=2)
    fitted = ParkingAnomalyDetector().fit([e for e in evs if e.label == "normal"])
    flags = fitted.predict(evs[:50])
    assert len(flags) == 50 and all(isinstance(f, Flag) for f in flags)
    assert all(f.pred in {"normal", "unauthorized", "fault", "anomaly"} for f in flags)


# ---- ② streaming ----
def test_streaming_alert_contract():
    from kev.streaming import StreamingMonitor, Alert
    from kev.occupancy import Event, simulate

    mon = StreamingMonitor()
    a = mon.alert_for(Event(3, 0, 100, False, 0, "unauthorized"), 0)
    assert isinstance(a, Alert) and a.kind == "unauthorized"
    assert a.time < 100 and a.lead > 0 and a.delay == mon.unauth_grace
    assert mon.alert_for(Event(3, 0, 100, True, 0, "normal"), 1) is None  # 등록 정상
    alerts = mon.run(simulate(n_events=400, seed=3))
    assert alerts and all(a.lead > 0 and a.delay >= 0 for a in alerts)


# ---- 정산 (billing) ----
def test_billing_settle_contract():
    from kev.billing import settle, Receipt, RATE, PENALTY
    from kev.occupancy import Event

    g = AnomalyCfg().start_grace_min
    r = settle(Event(1, 0, 7, True, 0, "normal"), "12가3456")
    assert isinstance(r, Receipt)
    assert r.amount == (7 - g) * RATE and r.saving > 0 and r.status == "정산완료"
    u = settle(Event(1, 0, 50, False, 0, "unauthorized"), "12가3456")
    assert u.penalty == PENALTY and u.amount == PENALTY
    miss = settle(Event(1, 0, 10, True, 0, "normal"), "")  # 번호판 미인식 표현
    assert miss.plate == "미인식"


# ---- 통합 파이프라인 계약(폰트·모델 비의존 경로) ----
def test_pipeline_record_contract_skip_path():
    """build_pipeline + on_frame/on_exit 통합 계약을 모델·폰트 없이 검증.
    motion=0 → skip → run_ocr False 이므로 PlateReader 없이도 동작."""
    from kev.pipeline import build_pipeline, Record
    from kev.occupancy import Event

    pipe = build_pipeline(plate_reader=None)  # adaptive_clf 없으면 룰 폴백
    rec = pipe.on_frame(_rand_bgr(seed=5), motion=0.0)
    assert isinstance(rec, Record) and rec.mode == "skip" and rec.run_ocr is False
    assert rec.plate is None and rec.anomaly is None
    rec = pipe.on_exit(rec, Event(1, 0, 40, False, 0, "unauthorized"))
    assert rec.anomaly in {"normal", "unauthorized", "fault", "anomaly"}
