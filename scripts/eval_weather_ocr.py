"""악천후가 번호판 OCR에 주는 영향 측정 — 적응 정책(악천후 시 보정/다중판독)의 근거.
clear / rain / fog / snow 각 조건에서 EasyOCR 정확도(완전일치·문자) 비교.
산출: data/weather_ocr.json, figs/weather_ocr.png
"""

from __future__ import annotations
import sys, json, random
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean

use_korean()
from kev.config import DATA, FIGS, SEED
from kev.plate_synth import render_plate, add_weather, random_plate_text
from kev.plate import char_accuracy

CONDS = ["clear", "rain", "fog", "snow"]


def main(n=40):
    import easyocr

    ocr = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    rng = np.random.default_rng(SEED)
    prng = random.Random(SEED)
    texts = [random_plate_text(prng) for _ in range(n)]

    res = {}
    for cond in CONDS:
        em = ca = 0
        for t in texts:
            img = render_plate(t)
            if cond != "clear":
                img = add_weather(img, cond, rng)
            pred = "".join(ocr.readtext(img, detail=0)).replace(" ", "")
            em += int(pred == t)
            ca += char_accuracy(pred, t)
        res[cond] = dict(exact=em / n, char=ca / n)

    (DATA / "weather_ocr.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    x = np.arange(len(CONDS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ex = [res[c]["exact"] * 100 for c in CONDS]
    ch = [res[c]["char"] * 100 for c in CONDS]
    ax.bar(x - w / 2, ex, w, label="완전일치", color="#2D3563")
    ax.bar(x + w / 2, ch, w, label="문자정확도", color="#E0962A")
    for i in range(len(CONDS)):
        ax.text(i - w / 2, ex[i], f"{ex[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, ch[i], f"{ch[i]:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["맑음", "비", "안개", "눈"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("%")
    ax.legend()
    ax.set_title("악천후별 번호판 OCR 정확도 — 적응 정책 필요성의 근거")
    fig.tight_layout()
    fig.savefig(FIGS / "weather_ocr.png", dpi=130)
    plt.close(fig)

    print("=== 악천후별 OCR ===")
    for c in CONDS:
        print(
            f"  {c:6s} 완전일치={res[c]['exact'] * 100:5.1f}%  문자정확도={res[c]['char'] * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
