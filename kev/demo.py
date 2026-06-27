"""통합 파이프라인 데모 시나리오 — ③→①→②를 한 입출차 흐름에 태움."""

from __future__ import annotations
import random
import numpy as np
import cv2

from .config import DATA, FIGS
from .plate_synth import make_scene, apply_env, random_plate_text, _asphalt
from .plate import OnnxYolo, PlateReader
from .occupancy import Event
from .pipeline import build_pipeline


# (이벤트 라벨, 환경, 모션, 차량유무, 설명)
SCENARIO = [
    ("normal", "day_normal", 0.35, True, "주간 정상 주차"),
    ("normal", "low_light", 0.30, True, "야간 — IR 폴백"),
    (None, "day_normal", 0.003, False, "차량 없음 — 절전"),
    ("normal", "backlit", 0.28, True, "역광 — 노출 보정"),
    ("normal", "rain", 0.30, True, "비 — 보정/다중판독"),
    ("normal", "fog", 0.26, True, "안개 — 대비 복원"),
    ("normal", "snow", 0.28, True, "눈 — 송이 가림"),
    ("unauthorized", "day_normal", 0.33, True, "무단주차(미등록)"),
    ("glare", "glare", 0.20, True, "글레어 — 판독 곤란"),
    ("fault_subtle", "day_normal", 0.05, True, "센서 약고장(등록 정상)"),
]


def _event(label: str, rng: random.Random) -> Event:
    """시나리오 라벨 → 점유 이벤트(데모용)."""
    start = rng.uniform(8 * 60, 18 * 60)
    if label == "normal":
        dur = rng.uniform(30, 60)
        return Event(1, start, start + dur, True, 0, "normal")
    if label == "unauthorized":
        dur = rng.uniform(30, 90)
        return Event(1, start, start + dur, False, 0, "unauthorized")
    if label == "fault_subtle":
        dur = rng.uniform(2.5, 4.0)
        return Event(1, start, start + dur, True, 0, "fault")
    # glare 등: 정상 점유(이상 아님)
    dur = rng.uniform(30, 60)
    return Event(1, start, start + dur, True, 0, "normal")


def default_plate_reader(gpu: bool = False) -> PlateReader:
    int8 = DATA / "runs" / "plate" / "weights" / "best_int8.onnx"
    det = OnnxYolo(int8, imgsz=320)
    return PlateReader(det, gpu=gpu)


def run_demo(save_fig: bool = True):
    rng = np.random.default_rng(7)
    prng = random.Random(7)
    reader = default_plate_reader(gpu=False)
    pipe = build_pipeline(reader)

    records, frames = [], []
    for ev_label, env, motion, has_car, desc in SCENARIO:
        if has_car:
            text = random_plate_text(prng)
            scene, _, _ = make_scene(text, rng)
            frame = apply_env(scene, env, rng)
        else:
            text = None
            frame = apply_env(_asphalt(640, 480, rng), env, rng)
        rec = pipe.on_frame(frame, motion)
        if ev_label is not None:
            lab = "normal" if ev_label == "glare" else ev_label
            pipe.on_exit(rec, _event(lab, prng))
        records.append((desc, text, rec))
        frames.append(frame)

    # 콘솔 표
    print(
        f"{'상황':18s} {'환경':11s} {'모드':9s} {'OCR':4s} "
        f"{'번호판(GT→pred)':22s} {'이상':12s} {'전력':5s}"
    )
    print("-" * 92)
    for desc, gt, rec in records:
        pred = rec.plate or "—"
        gtp = f"{gt or '—'}→{pred}"
        v = "✓" if rec.plate_valid else (" " if rec.plate is None else "✗")
        print(
            f"{desc:18s} {rec.env:11s} {rec.mode:9s} {str(rec.run_ocr):4s} "
            f"{gtp:22s} {str(rec.anomaly or '—'):12s} {rec.power:4.2f} {v}"
        )

    if save_fig:
        _save_demo_figure(records, frames)
    return records


def _save_demo_figure(records, frames):
    """데모 결과를 격자 그림으로 저장(figs/pipeline_demo.png)."""
    import matplotlib

    matplotlib.use("Agg")
    from .plotting import use_korean

    use_korean()
    import matplotlib.pyplot as plt

    ncol = 4
    nrow = (len(records) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3 * nrow))
    axes = axes.ravel()
    for ax in axes[len(records) :]:
        ax.axis("off")
    for ax, (desc, gt, rec), fr in zip(axes, records, frames):
        ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        tag = f"{desc}\n③{rec.env}·{rec.mode}"
        if rec.run_ocr:
            tag += f"\n①{rec.plate or '검출X'}"
        if rec.anomaly:
            tag += f"\n②{rec.anomaly}"
        color = "#C0392B" if rec.anomaly not in (None, "normal") else "#1E5631"
        ax.set_title(tag, fontsize=8, color=color)
    fig.suptitle(
        "코랏 카스토퍼 엣지 파이프라인 — ③환경적응 → ①번호판 → ②이상탐지",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "pipeline_demo.png", dpi=130)
    plt.close(fig)
