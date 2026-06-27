"""③·② 시각화.
  adaptive_decisions.png : ③가 각 환경 프레임에서 내린 판단(환경→모드→OCR)
  anomaly_sessions.png   : ②가 보는 데이터(점유 타임라인 + 세션 산점도)
"""
from __future__ import annotations
import sys, random
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean; use_korean()
from kev.config import FIGS, DATA, SEED, ENVS, AnomalyCfg
from kev.plate_synth import make_scene, apply_env, random_plate_text
from kev.adaptive import AdaptiveSensor
from kev.occupancy import simulate
from kev.anomaly import ParkingAnomalyDetector

ENV_KO = {"day_normal": "주간정상", "low_light": "야간", "glare": "글레어",
          "backlit": "역광", "overexposed": "과노출", "rain": "비",
          "fog": "안개", "snow": "눈"}
MODE_KO = {"rgb_full": "RGB풀", "rgb_boost": "RGB보정", "ir": "IR폴백", "skip": "절전"}


# ===== ③ 판단 시각화 =====
def adaptive_decisions():
    rng = np.random.default_rng(11); prng = random.Random(11)
    clf = joblib.load(DATA / "adaptive_clf.joblib")
    sensor = AdaptiveSensor(classifier=clf)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, env in zip(axes.ravel(), ENVS):
        scene, _, _ = make_scene(random_plate_text(prng), rng)
        motion = 0.3
        img = apply_env(scene, env, rng)
        d = sensor.step(img, motion=motion)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ok = "✓" if d.env == env else "✗"
        col = "#1E5631" if d.env == env else "#C0392B"
        ax.set_title(f"실제:{ENV_KO[env]} → 판단:{ENV_KO.get(d.env, d.env)} {ok}\n"
                     f"모드 {MODE_KO[d.mode]} · OCR {'O' if d.run_ocr else 'X'} · "
                     f"판독성 {d.readable:.2f}", fontsize=9, color=col)
    fig.suptitle("③ 휘도·악천후 환경적응 — 각 환경에서 센서가 내린 판단",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "adaptive_decisions.png", dpi=130)
    plt.close(fig)
    print("saved figs/adaptive_decisions.png")


# ===== ② 데이터 시각화 =====
LAB_COLOR = {"normal": "#2E8B4E", "unauthorized": "#E0962A", "fault": "#C0392B"}
LAB_KO = {"normal": "정상", "unauthorized": "무단점유(미등록)", "fault": "센서고장"}


def anomaly_sessions():
    evs = simulate(n_events=1400, seed=SEED)
    normal = [e for e in evs if e.label == "normal"]
    det = ParkingAnomalyDetector(AnomalyCfg()).fit(normal)
    flags = det.predict(evs)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    # (좌) 점유 타임라인 — 하루 창, 스팟 0~7
    win = 24 * 60
    rows = [(e, f) for e, f in zip(evs, flags)
            if e.spot < 8 and e.start < win and (e.end - e.start) < 8 * 60]
    for e, f in rows:
        flagged = f.pred != "normal"
        ax[0].barh(e.spot, (e.end - e.start) / 60, left=e.start / 60, height=0.6,
                   color=LAB_COLOR[e.label], alpha=0.85,
                   edgecolor=("black" if flagged else "none"),
                   linewidth=1.6 if flagged else 0)
    ax[0].set_xlabel("시각 (시)"); ax[0].set_ylabel("주차면 #")
    ax[0].set_xlim(0, 24); ax[0].set_yticks(range(8))
    ax[0].set_title("점유 타임라인 (검은 테두리=이상 탐지)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LAB_COLOR.values()]
    ax[0].legend(handles, [LAB_KO[k] for k in LAB_COLOR], fontsize=8, loc="upper right")

    # (우) 세션 산점도 — 체류시간(log) × flicker, 색=라벨, 테두리=탐지
    for e, f in zip(evs, flags):
        dur = max(0.2, e.end - e.start)
        jit = e.flicker + np.random.uniform(-0.3, 0.3)
        flagged = f.pred != "normal"
        ax[1].scatter(dur, jit, s=22, c=LAB_COLOR[e.label], alpha=0.6,
                      edgecolors=("black" if flagged else "none"), linewidths=0.7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("체류시간 (분, log)")
    ax[1].set_ylabel("센서 토글수 (flicker)")
    ax[1].set_title("세션 분포 — 고장(짧음/김/토글)이 정상과 분리")
    ax[1].axvline(2.0, ls="--", c="#888", lw=1); ax[1].axvline(18*60, ls="--", c="#888", lw=1)
    ax[1].axhline(20, ls="--", c="#888", lw=1)

    fig.suptitle("② 불법주차 이상탐지 — 데이터(이미지 아님): 점유×예약 세션",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "anomaly_sessions.png", dpi=130)
    plt.close(fig)
    print("saved figs/anomaly_sessions.png")


if __name__ == "__main__":
    adaptive_decisions()
    anomaly_sessions()
