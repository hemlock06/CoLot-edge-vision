"""kev 단위 테스트 — 학습/ocr 없이 빠르게 도는 핵심 검증."""
import re
import random
import numpy as np
import pytest

from kev.config import PLATE_RE, ENVS, WEATHER_ENVS, DATA
from kev.plate_synth import (render_plate, random_plate_text, make_scene, relight,
                             add_weather, apply_env)
from kev.adaptive import (brightness_features, FEATURE_ORDER, feature_vector,
                          rule_label, readability, AdaptiveSensor)
from kev.plate import preprocess, char_accuracy, _letterbox
from kev.occupancy import simulate, Event
from kev.anomaly import ParkingAnomalyDetector, features, FEATS


rng = np.random.default_rng(0)
prng = random.Random(0)


# ---- 합성 데이터 ----
def test_plate_text_format():
    for _ in range(50):
        assert re.match(PLATE_RE, random_plate_text(prng))


def test_render_plate_shape():
    img = render_plate("12가3456")
    assert img.shape == (110, 520, 3) and img.dtype == np.uint8


def test_make_scene_bbox_in_bounds():
    scene, (x0, y0, x1, y1), text = make_scene("12가3456", rng, size=(640, 480))
    assert scene.shape == (480, 640, 3)
    assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480


# ---- ③ adaptive ----
def test_brightness_features_keys():
    f = brightness_features(make_scene("12가3456", rng)[0])
    assert set(FEATURE_ORDER).issubset(f) and len(feature_vector(f)) == len(FEATURE_ORDER)


def test_relight_changes_brightness():
    scene, _, _ = make_scene("12가3456", rng)
    dark = brightness_features(relight(scene, "low_light", rng))["mean"]
    over = brightness_features(relight(scene, "overexposed", rng))["mean"]
    assert dark < over


@pytest.mark.parametrize("env", ENVS)
def test_readability_range(env):
    img = apply_env(make_scene("12가3456", rng)[0], env, rng)
    assert 0.0 <= readability(brightness_features(img)) <= 1.0


@pytest.mark.parametrize("w", WEATHER_ENVS)
def test_weather_changes_image(w):
    scene = make_scene("12가3456", rng)[0]
    out = add_weather(scene, w, rng)
    assert out.shape == scene.shape and out.dtype == np.uint8
    assert not np.array_equal(out, scene)


def test_fog_raises_dark_channel():
    scene = make_scene("12가3456", rng)[0]
    base = brightness_features(scene)["dark_channel"]
    fog = brightness_features(add_weather(scene, "fog", rng))["dark_channel"]
    assert fog > base                       # 안개는 dark-channel을 높임(haze)


def test_adaptive_classifies_weather():
    import joblib
    p = DATA / "adaptive_clf.joblib"
    if not p.exists():
        pytest.skip("분류기 없음 (scripts/eval_adaptive.py 먼저)")
    clf = joblib.load(p)
    sensor = AdaptiveSensor(classifier=clf)
    hit = 0
    for w in WEATHER_ENVS:
        for _ in range(5):
            img = add_weather(make_scene(random_plate_text(prng), rng)[0], w, rng)
            hit += sensor.step(img, motion=0.3).env == w
    assert hit >= 11        # 15개 중 11+ (악천후 분류 동작)


def test_adaptive_skip_on_no_motion():
    img = make_scene("12가3456", rng)[0]
    assert AdaptiveSensor().step(img, motion=0.0).mode == "skip"


# ---- ① plate ----
def test_preprocess_shape():
    x, r, dx, dy = preprocess(np.zeros((480, 640, 3), np.uint8), 320)
    assert x.shape == (1, 3, 320, 320) and x.dtype == np.float32
    assert 0.0 < r <= 1.0


def test_char_accuracy():
    assert char_accuracy("12가3456", "12가3456") == 1.0
    assert char_accuracy("12가3456", "12가3457") == pytest.approx(6/7, abs=1e-6)
    assert char_accuracy("", "12가3456") == 0.0


def test_onnx_detector_finds_plate():
    int8 = DATA / "runs" / "plate" / "weights" / "best_int8.onnx"
    if not int8.exists():
        pytest.skip("INT8 모델 없음 (scripts/build_plate.py 먼저)")
    from kev.plate import OnnxYolo

    def iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix1-ix0) * max(0, iy1-iy0)
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter/ua if ua > 0 else 0
    det = OnnxYolo(int8, imgsz=320)
    hit = 0
    for _ in range(8):
        scene, gt, _ = make_scene(random_plate_text(prng), rng)
        dets = det.detect(scene)
        if dets and max(iou(d[0], gt) for d in dets) >= 0.5:
            hit += 1
    assert hit >= 6        # 8개 중 6개 이상 검출


# ---- ② anomaly ----
def test_simulate_labels():
    evs = simulate(n_events=400, seed=1)
    labs = {e.label for e in evs}
    assert {"normal", "unauthorized", "overstay", "fault"}.issubset(labs)
    assert len(evs) == 400


def test_rule_catches_violations():
    det = ParkingAnomalyDetector()
    # 무단(예약 없음)
    assert det.rule(Event(1, 0, 40, False, -1, 0, "unauthorized")) == "unauthorized"
    # 초과(결제시간 초과)
    assert det.rule(Event(1, 0, 100, True, 40, 0, "overstay")) == "overstay"
    # 명백 고장(flicker)
    assert det.rule(Event(1, 0, 30, True, 100, 40, "fault")) == "sensor_fault"
    # 정상
    assert det.rule(Event(1, 0, 40, True, 80, 0, "normal")) is None


def test_detector_fit_predict():
    evs = simulate(n_events=600, seed=2)
    normal = [e for e in evs if e.label == "normal"]
    det = ParkingAnomalyDetector().fit(normal)
    flags = det.predict(evs[:100])
    assert len(flags) == 100
    assert all(f.pred in {"normal", "unauthorized", "overstay", "fault", "anomaly"}
               for f in flags)


def test_features_dim():
    e = Event(1, 0, 40, True, 80, 0, "normal")
    assert len(features(e)) == len(FEATS)
