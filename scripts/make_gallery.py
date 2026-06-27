"""합성 이미지 갤러리 — 번호판 / 조명 5종 / 악천후 4종을 한 장에.
산출: figs/gallery.png
"""
from __future__ import annotations
import sys, random
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean; use_korean()
from kev.config import FIGS, LIGHT_ENVS
from kev.plate_synth import (render_plate, make_scene, relight, add_weather,
                             random_plate_text, _asphalt)

rng = np.random.default_rng(3); prng = random.Random(3)
LIGHT_KO = {"day_normal": "주간 정상", "low_light": "야간/저조도", "glare": "글레어",
            "backlit": "역광", "overexposed": "과노출"}
WEATHER = [("clear", "맑음"), ("rain", "비"), ("fog", "안개"), ("snow", "눈")]

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 5, hspace=0.32, wspace=0.08)


def show(ax, img, title, color="#1F2A3A"):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.set_title(title, fontsize=9, color=color)


# Row A — 깨끗한 합성 한글 번호판
for j in range(5):
    t = random_plate_text(prng)
    show(fig.add_subplot(gs[0, j]), render_plate(t), t)
fig.text(0.012, 0.80, "① 합성 번호판", rotation=90, va="center",
         fontsize=11, fontweight="bold", color="#2D3563")

# Row B — 동일 장면, 조명 5종
scene, bbox, gt = make_scene(random_plate_text(prng), rng)
for j, env in enumerate(LIGHT_ENVS):
    img = relight(scene, env, rng)
    show(fig.add_subplot(gs[1, j]), img, f"{LIGHT_KO[env]}\n({env})")
fig.text(0.012, 0.50, "② 조명 환경 5종", rotation=90, va="center",
         fontsize=11, fontweight="bold", color="#2D3563")

# Row C — 동일 장면, 악천후 + 빈 노면(절전)
scene2, _, _ = make_scene(random_plate_text(prng), rng)
for j, (w, ko) in enumerate(WEATHER):
    img = scene2.copy() if w == "clear" else add_weather(relight(scene2, "day_normal", rng), w, rng)
    show(fig.add_subplot(gs[2, j]), img, ko, color=("#1E5631" if w == "clear" else "#C0392B"))
show(fig.add_subplot(gs[2, 4]), relight(_asphalt(640, 480, rng), "day_normal", rng),
     "빈 노면(절전)", color="#6B7280")
fig.text(0.012, 0.20, "③ 악천후 3종", rotation=90, va="center",
         fontsize=11, fontweight="bold", color="#2D3563")

fig.suptitle("CoLot-edge 합성 시뮬레이션 이미지 — 번호판 · 조명 · 악천후",
             fontsize=14, fontweight="bold", y=0.96)
fig.savefig(FIGS / "gallery.png", dpi=130, bbox_inches="tight")
print("saved figs/gallery.png")
