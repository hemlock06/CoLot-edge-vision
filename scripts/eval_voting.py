"""#1 평가 — 단일 프레임 OCR vs 다중프레임 투표 (조건별).

★ 정직성(A3 수정): 실제 한 차량의 연속 프레임은 *같은 악천후 조건*을 공유(상관 노이즈)한다.
따라서 트랙당 악천후를 **1회만** 실현(고정 view)하고, 프레임별로는 **독립 센서 노이즈**만 더한다.
이는 보수적(하한) 모델 — 투표가 회복하는 건 센서 노이즈 부분뿐이라 이득이 정직하게 작아진다.
(이전엔 프레임마다 새 악천후를 줘 오류 독립성을 과대평가 → 투표 이득 과대)
산출: data/voting_metrics.json, figs/voting.png
"""
from __future__ import annotations
import sys, json, random
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean; use_korean()
from kev.config import DATA, FIGS, SEED
from kev.plate_synth import render_plate, add_weather, random_plate_text
from kev.plate import correct_plate, PLATE_ALLOW
from kev.tracking import PlateVoter

CONDS = [("clear", "맑음"), ("rain", "비"), ("fog", "안개"), ("snow", "눈")]


def main(n_veh=40, k_frames=7, noise_std=12.0):
    import easyocr
    ocr = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    rng = np.random.default_rng(SEED); prng = random.Random(SEED)

    res = {}
    for cond, ko in CONDS:
        single_frac = voted_hits = 0.0
        for _ in range(n_veh):
            t = random_plate_text(prng)
            base = render_plate(t)
            if cond != "clear":
                base = add_weather(base, cond, rng)      # 트랙당 1회 (조건 공유)
            voter = PlateVoter(); frame_ok = 0
            for _ in range(k_frames):
                # 프레임별 독립 센서 노이즈만 (악천후 조건은 고정 → 상관)
                fr = np.clip(base.astype(np.float32) + rng.normal(0, noise_std, base.shape),
                             0, 255).astype(np.uint8)
                if rng.random() < 0.35:
                    fr = cv2.GaussianBlur(fr, (3, 3), 0)
                raw = "".join(ocr.readtext(fr, detail=0, allowlist=PLATE_ALLOW))
                pred = correct_plate(raw)[0]
                voter.add(pred, already_corrected=True)
                frame_ok += int(pred == t)
            single_frac += frame_ok / k_frames
            voted_hits += int(voter.consensus()["text"] == t)
        res[cond] = dict(single=single_frac / n_veh, voted=voted_hits / n_veh)

    res["_model"] = "conservative: fixed-weather per track + independent sensor noise"
    (DATA / "voting_metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    cnd = [c for c, _ in CONDS]
    x = np.arange(len(cnd)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    s = [res[c]["single"] * 100 for c in cnd]
    v = [res[c]["voted"] * 100 for c in cnd]
    ax.bar(x - w/2, s, w, label="단일 프레임", color="#C0C6CE")
    ax.bar(x + w/2, v, w, label=f"{k_frames}프레임 투표", color="#2D6CB5")
    for i in range(len(cnd)):
        ax.text(i - w/2, s[i], f"{s[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, v[i], f"{v[i]:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([ko for _, ko in CONDS])
    ax.set_ylim(0, 108); ax.set_ylabel("완전일치 (%)"); ax.legend()
    ax.set_title("다중프레임 투표 (보수적: 조건 공유 + 센서노이즈만 독립)")
    fig.tight_layout(); fig.savefig(FIGS / "voting.png", dpi=130); plt.close(fig)

    print("=== 단일 → 투표 (완전일치, 보수적 모델) ===")
    for c, ko in CONDS:
        print(f"  {ko:4s} {res[c]['single']*100:5.1f}% → {res[c]['voted']*100:5.1f}%")


if __name__ == "__main__":
    main()
