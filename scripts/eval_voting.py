"""#1 평가 — 단일 프레임 OCR vs 다중프레임 투표 (조건별).
산출: data/voting_metrics.json, figs/voting.png
"""
from __future__ import annotations
import sys, json, random
from pathlib import Path
import numpy as np
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


def main(n_veh=30, k_frames=7):
    import easyocr
    ocr = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    rng = np.random.default_rng(SEED); prng = random.Random(SEED)

    res = {}
    for cond, ko in CONDS:
        single_hits = voted_hits = 0
        single_frac = 0.0
        for _ in range(n_veh):
            t = random_plate_text(prng)
            voter = PlateVoter()
            frame_ok = 0
            for _ in range(k_frames):
                img = render_plate(t)
                if cond != "clear":
                    img = add_weather(img, cond, rng)
                raw = "".join(ocr.readtext(img, detail=0, allowlist=PLATE_ALLOW))
                pred = correct_plate(raw)[0]
                voter.add(pred, already_corrected=True)
                frame_ok += int(pred == t)
            single_frac += frame_ok / k_frames           # 기대 단일프레임 정확도
            voted_hits += int(voter.consensus()["text"] == t)
        res[cond] = dict(single=single_frac / n_veh, voted=voted_hits / n_veh)

    (DATA / "voting_metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    x = np.arange(len(CONDS)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    s = [res[c]["single"] * 100 for c, _ in CONDS]
    v = [res[c]["voted"] * 100 for c, _ in CONDS]
    ax.bar(x - w/2, s, w, label="단일 프레임", color="#C0C6CE")
    ax.bar(x + w/2, v, w, label=f"{7}프레임 투표", color="#2D6CB5")
    for i in range(len(CONDS)):
        ax.text(i - w/2, s[i], f"{s[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, v[i], f"{v[i]:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([ko for _, ko in CONDS])
    ax.set_ylim(0, 108); ax.set_ylabel("완전일치 (%)"); ax.legend()
    ax.set_title("다중프레임 투표가 악천후 OCR을 회복 (단일 vs 투표)")
    fig.tight_layout(); fig.savefig(FIGS / "voting.png", dpi=130); plt.close(fig)

    print("=== 단일 → 투표 (완전일치) ===")
    for c, ko in CONDS:
        print(f"  {ko:4s} {res[c]['single']*100:5.1f}% → {res[c]['voted']*100:5.1f}%")


if __name__ == "__main__":
    main()
