# -*- coding: utf-8 -*-
"""③ 환경분류 과적합 면밀 검토.
- A↔B 양방향 OOD (대칭이면 일반화, 한쪽만 높으면 과적합)
- 5-fold CV (분포 내 일반화)
- 클래스별 in-dist−OOD 격차 (격차 큰 클래스 = 과적합 의심)
- 피처 중요도 + 피처 ablation (streak_coh가 OOD 비를 진짜 돕나)
산출: data/overfit_audit.txt (UTF-8)
"""

from __future__ import annotations
import sys, io, random
from pathlib import Path
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.config import ENVS, SEED, DATA
from kev.plate_synth import make_scene, apply_env, random_plate_text
from kev.adaptive import brightness_features, feature_vector, FEATURE_ORDER

OUT = []


def log(s=""):
    OUT.append(str(s))


def gen(n, seed, variant, drop=None):
    g = np.random.default_rng(seed)
    p = random.Random(seed)
    X, y = [], []
    keep = [i for i, f in enumerate(FEATURE_ORDER) if f not in (drop or [])]
    for _ in range(n):
        sc = make_scene(random_plate_text(p), g)[0]
        for env in ENVS:
            im = apply_env(sc, env, g, variant=variant)
            if g.random() < 0.5:
                im = cv2.GaussianBlur(im, (3, 3), 0)
            im = np.clip(
                im.astype(np.float32) + g.normal(0, 9, im.shape), 0, 255
            ).astype(np.uint8)
            X.append(feature_vector(brightness_features(im))[keep])
            y.append(env)
    return np.array(X, np.float32), np.array(y)


def rf(seed=0):
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1
    )


# ── 데이터 (장면 단위 분리) ──
Xa, ya = gen(80, SEED, "A")
XaT, yaT = gen(35, SEED + 777, "A")  # A train / A test
Xb, yb = gen(80, SEED + 5, "B")
XbT, ybT = gen(35, SEED + 777, "B")  # B train / B test

# ── (1) 양방향 OOD ──
clfA = rf().fit(Xa, ya)
clfB = rf().fit(Xb, yb)
log("=== (1) 양방향 OOD (대칭=일반화 / 비대칭=과적합) ===")
log(f"  A학습 → A테스트(in-dist) : {accuracy_score(yaT, clfA.predict(XaT)):.3f}")
log(f"  A학습 → B테스트(OOD)     : {accuracy_score(ybT, clfA.predict(XbT)):.3f}")
log(f"  B학습 → B테스트(in-dist) : {accuracy_score(ybT, clfB.predict(XbT)):.3f}")
log(f"  B학습 → A테스트(역OOD)   : {accuracy_score(yaT, clfB.predict(XaT)):.3f}")

# ── (2) 5-fold CV (분포 내 일반화) ──
cv = cross_val_score(rf(), Xa, ya, cv=5)
log("\n=== (2) 5-fold CV (A) ===")
log(f"  CV acc {cv.mean():.3f} ± {cv.std():.3f}  → in-dist 단일(=학습셋 아님) 과 비교")

# ── (3) 클래스별 in-dist−OOD 격차 ──
log("\n=== (3) 클래스별 재현율 (A→A / A→B) + 격차 ===")
ra = classification_report(
    yaT, clfA.predict(XaT), labels=ENVS, output_dict=True, zero_division=0
)
rb = classification_report(
    ybT, clfA.predict(XbT), labels=ENVS, output_dict=True, zero_division=0
)
for e in ENVS:
    gap = ra[e]["recall"] - rb[e]["recall"]
    flag = "  ⚠과적합 의심" if gap > 0.12 else ""
    log(
        f"  {e:12s} in-dist {ra[e]['recall']:.2f}  OOD {rb[e]['recall']:.2f}  격차 {gap:+.2f}{flag}"
    )

# ── (4) 피처 중요도 (날씨 3종 위치) ──
log("\n=== (4) 피처 중요도 top8 ===")
imp = sorted(zip(FEATURE_ORDER, clfA.feature_importances_), key=lambda t: -t[1])
for f, v in imp[:8]:
    log(f"  {f:14s} {v:.3f}")
for f in ("speckle", "streak_coh"):
    log(f"  · {f}: {dict(imp)[f]:.3f}")

# ── (5) 피처 ablation — OOD '비' 재현율 ──
log("\n=== (5) Ablation: OOD '비' 재현율 (피처 제거 시) ===")
for name, drop in [
    ("전체 15피처", []),
    ("− streak_coh(각도불변)", ["streak_coh"]),
    ("− speckle", ["speckle"]),
    ("기본 13(날씨피처 0)", ["speckle", "streak_coh"]),
]:
    Xa2, ya2 = gen(80, SEED, "A", drop=drop)
    XbT2, ybT2 = gen(35, SEED + 777, "B", drop=drop)
    c = rf().fit(Xa2, ya2)
    rep = classification_report(
        ybT2, c.predict(XbT2), labels=ENVS, output_dict=True, zero_division=0
    )
    log(
        f"  {name:22s} 비 OOD 재현율 {rep['rain']['recall']:.2f} · 전체 OOD {accuracy_score(ybT2, c.predict(XbT2)):.3f}"
    )

io.open(str(DATA / "overfit_audit.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n".join(OUT))
