"""① 번호판 파이프라인 빌드+평가: 학습 → ONNX → INT8 → 벤치 → OCR.
산출: data/plate_bench.json, figs/plate_quant.png, figs/plate_samples.png
"""
from __future__ import annotations
import sys, os, json, time, csv
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean; use_korean()
from kev.config import DATA, FIGS, SEED, PlateCfg
from kev.plate import (build_yolo_dataset, train_detector, export_onnx,
                       quantize_int8, quantize_static_int8, OnnxYolo,
                       PlateReader, char_accuracy)

DS = DATA / "plate_ds"
IMGSZ = 320


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_val_gt():
    rows = []
    with open(DS / "val_gt.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["file"], (int(r["x0"]), int(r["y0"]),
                         int(r["x1"]), int(r["y1"])), r["text"]))
    return rows


def det_rate_onnx(model: OnnxYolo, val):
    hit = 0
    for fn, gt, _ in val:
        img = cv2.imread(str(DS / "images" / "val" / fn))
        dets = model.detect(img)
        if dets and max(iou(d[0], gt) for d in dets) >= 0.5:
            hit += 1
    return hit / len(val)


def det_rate_torch(model, val):
    hit = 0
    for fn, gt, _ in val:
        img = cv2.imread(str(DS / "images" / "val" / fn))
        r = model(img, verbose=False)[0]
        boxes = [b.xyxy[0].tolist() for b in r.boxes]
        if boxes and max(iou(b, gt) for b in boxes) >= 0.5:
            hit += 1
    return hit / len(val)


def torch_latency(model, img, n=30, warmup=5, device=0):
    for _ in range(warmup):
        model(img, verbose=False, device=device)
    t = time.perf_counter()
    for _ in range(n):
        model(img, verbose=False, device=device)
    return (time.perf_counter() - t) / n * 1000.0


def ocr_eval(reader, sub):
    em = ca = 0
    for fn, gt, text in sub:
        img = cv2.imread(str(DS / "images" / "val" / fn))
        res = reader.read(img)
        pred = max(res, key=lambda r: r["conf"])["text"] if res else ""
        em += int(pred == text); ca += char_accuracy(pred, text)
    return em / len(sub), ca / len(sub)


def main():
    import gc, torch
    from ultralytics import YOLO
    best = DATA / "runs" / "plate" / "weights" / "best.pt"
    # 1) 데이터셋 (없으면 생성)
    if not (DS / "val_gt.csv").exists():
        print("[1/5] 합성 YOLO 데이터셋...")
        build_yolo_dataset(n_train=320, n_val=90, root=DS, seed=SEED)
    val = load_val_gt()
    sample = cv2.imread(str(DS / "images" / "val" / val[0][0]))

    # 2) 학습 (best.pt 있으면 재사용)
    if not best.exists():
        print("[2/5] YOLOv8n 학습...")
        best = train_detector(DS / "data.yaml", epochs=40, imgsz=IMGSZ)
    else:
        print("[2/5] 기존 best.pt 재사용:", best)

    # 3) export + 정적 INT8 양자화(캘리브레이션)
    print("[3/5] ONNX export + 정적 INT8...")
    onnx_fp32 = export_onnx(best, imgsz=IMGSZ)
    calib = [DS / "images" / "val" / fn for fn, _, _ in val[:50]]
    onnx_int8 = quantize_static_int8(onnx_fp32, calib, imgsz=IMGSZ)
    size_mb = lambda p: os.path.getsize(p) / 1e6

    # 4) 벤치 — torch(GPU/CPU) 먼저, 끝나면 GPU 메모리 해제
    print("[4/5] 벤치마크 (torch)...")
    yolo = YOLO(str(best))
    variants = {}
    variants["PyTorch GPU\n(클라우드 FP32)"] = dict(
        size=size_mb(best), lat=torch_latency(yolo, sample, device=0),
        det=det_rate_torch(yolo, val), kind="cloud")
    variants["PyTorch CPU\n(FP32)"] = dict(
        size=size_mb(best), lat=torch_latency(yolo, sample, device="cpu"),
        det=None, kind="mid")

    # 5) OCR — easyocr 1개(CPU) 공유. 클라우드 검출(GPU yolo) OCR 먼저.
    print("[5/5] OCR end-to-end...")
    import easyocr
    ocr = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    sub = val[:60]
    cloud_em, cloud_ca = ocr_eval(PlateReader(yolo, ocr_reader=ocr), sub)

    del yolo; gc.collect(); torch.cuda.empty_cache()   # GPU 해제 후 onnx 단계

    print("       벤치마크 (onnx CPU)...")
    fp32 = OnnxYolo(onnx_fp32, imgsz=IMGSZ)
    int8 = OnnxYolo(onnx_int8, imgsz=IMGSZ)
    variants["ONNX CPU\n(FP32)"] = dict(
        size=size_mb(onnx_fp32), lat=fp32.latency_ms(sample),
        det=det_rate_onnx(fp32, val), kind="mid")
    variants["ONNX CPU INT8\n(온디바이스 엣지)"] = dict(
        size=size_mb(onnx_int8), lat=int8.latency_ms(sample),
        det=det_rate_onnx(int8, val), kind="edge")

    edge_em, edge_ca = ocr_eval(PlateReader(int8, ocr_reader=ocr), sub)

    bench = {k: {kk: vv for kk, vv in v.items()} for k, v in variants.items()}
    bench["ocr"] = dict(cloud_exact=cloud_em, cloud_char=cloud_ca,
                        edge_exact=edge_em, edge_char=edge_ca, n=len(sub))
    (DATA / "plate_bench.json").write_text(
        json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 그림: 크기 / 지연 ---
    names = list(variants)
    colors = {"cloud": "#9aa1ac", "mid": "#C0C6CE", "edge": "#2E8B4E"}
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax[0].bar(names, [variants[n]["size"] for n in names],
              color=[colors[variants[n]["kind"]] for n in names])
    ax[0].set_title("모델 크기 (MB)"); ax[0].tick_params(axis="x", labelsize=7.5)
    ax[1].bar(names, [variants[n]["lat"] for n in names],
              color=[colors[variants[n]["kind"]] for n in names])
    ax[1].set_title("추론 지연 (ms/img)"); ax[1].tick_params(axis="x", labelsize=7.5)
    for a in ax:
        for i, n in enumerate(names):
            v = variants[n]["size"] if a is ax[0] else variants[n]["lat"]
            a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("번호판 검출 — 풀클라우드 → 온디바이스 경량화", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "plate_quant.png", dpi=130); plt.close(fig)

    # --- 그림: 검출+OCR 샘플 ---
    reader_edge = PlateReader(int8, ocr_reader=ocr)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for ax_, (fn, gt, text) in zip(axes.ravel(), sub[:6]):
        img = cv2.imread(str(DS / "images" / "val" / fn))
        res = reader_edge.read(img)
        for r in res:
            x0, y0, x1, y1 = r["bbox"]
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 200, 80), 2)
        pred = max(res, key=lambda r: r["conf"])["text"] if res else "—"
        ax_.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax_.axis("off")
        ok = "✓" if pred == text else "✗"
        ax_.set_title(f"GT {text} / pred {pred} {ok}", fontsize=8.5)
    fig.suptitle("온디바이스(INT8) 검출 + 한글 OCR 샘플", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "plate_samples.png", dpi=130); plt.close(fig)

    print("\n=== 경량화 벤치 ===")
    for n in names:
        v = variants[n]
        d = f"det={v['det']:.2f}" if v["det"] is not None else "det=  - "
        print(f"  {n.splitlines()[0]:16s} size={v['size']:5.1f}MB  lat={v['lat']:6.1f}ms  {d}")
    print("=== OCR e2e ===")
    print(f"  클라우드(GPU): exact={cloud_em:.2f} char={cloud_ca:.3f}")
    print(f"  엣지(INT8 CPU): exact={edge_em:.2f} char={edge_ca:.3f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
