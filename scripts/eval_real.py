"""실데이터 검증 — 합성으로 학습/구성한 파이프라인을 실 차량·번호판 이미지에 적용.
(HF keremberke/license-plate-object-detection 실 이미지)
정직 측정: ③환경분류(합성→실 OOD) · ①검출(합성 도메인 갭) · EasyOCR(실 판독).
산출: data/real_metrics.json, figs/real_validation.png
"""

from __future__ import annotations
import sys, json, glob
from pathlib import Path
import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean

use_korean()
from kev.config import DATA, FIGS
from kev.adaptive import AdaptiveSensor
from kev.plate import OnnxYolo, PLATE_ALLOW
import joblib


def main():
    import easyocr

    imgs = sorted(glob.glob(str(DATA / "real_test" / "hf_*.jpg")))
    clf = joblib.load(DATA / "adaptive_clf.joblib")
    sensor = AdaptiveSensor(classifier=clf)
    det = OnnxYolo(
        DATA / "runs" / "plate" / "weights" / "best_int8.onnx", imgsz=320, conf=0.25
    )
    ocr = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

    rows, envs, det_hit, ocr_any = [], {}, 0, 0
    for f in imgs:
        im = cv2.imread(f)
        env = sensor.step(im, motion=0.3).env
        envs[env] = envs.get(env, 0) + 1
        nbox = len(det.detect(im))  # 합성 검출기 (도메인 갭)
        det_hit += int(nbox > 0)
        texts = ocr.readtext(im, detail=0, allowlist=PLATE_ALLOW)
        best = max(("".join(t.split()) for t in texts), key=len, default="")
        ocr_any += int(len(best) >= 4)
        rows.append(dict(file=Path(f).name, env=env, nbox=nbox, ocr=best))

    res = dict(
        n=len(imgs),
        det_hit_rate=det_hit / max(1, len(imgs)),  # 합성 검출기 실 이미지 검출율
        ocr_read_rate=ocr_any / max(1, len(imgs)),  # EasyOCR 실 판독율(≥4자)
        env_dist=envs,
        rows=rows,
    )
    (DATA / "real_metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 몽타주
    n = min(12, len(imgs))
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow))
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), rows[:n]):
        im = cv2.imread(str(DATA / "real_test" / r["file"]))
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(
            f"③{r['env']} · 검출 {r['nbox']} · OCR:{r['ocr'] or '—'}", fontsize=8
        )
    fig.suptitle(
        "실데이터 검증 — 합성 학습 파이프라인을 실 번호판 이미지에 적용",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "real_validation.png", dpi=130)
    plt.close(fig)

    print(f"=== 실데이터 {res['n']}장 ===")
    print(f"  ③ 환경분포(실 photo): {envs}")
    print(f"  ① 합성 검출기 검출율: {res['det_hit_rate'] * 100:.0f}%  ← 도메인 갭")
    print(f"  EasyOCR 실 판독율(≥4자): {res['ocr_read_rate'] * 100:.0f}%")
    print("  OCR 예시:", [r["ocr"] for r in rows if r["ocr"]][:6])


if __name__ == "__main__":
    main()
