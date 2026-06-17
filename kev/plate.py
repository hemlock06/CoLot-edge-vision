"""① 번호판 검출 + OCR + 온디바이스 경량화.

핵심 PM 기여(코랏): 풀클라우드(GPU FP32) → 온디바이스(CPU INT8) 전환.
  - YOLOv8n을 합성 한글 번호판으로 학습 → 검출기
  - ONNX export → INT8 동적 양자화
  - 클라우드/엣지 백엔드의 크기·지연·검출정확도 비교
  - easyocr 한글 인식 + 한국 번호판 포맷 검증
"""
from __future__ import annotations
import re, time, csv
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from .config import PlateCfg, PLATE_RE, DATA, SEED
from .plate_synth import make_scene, relight, random_plate_text


# ---- 합성 YOLO 데이터셋 -------------------------------------------------
def build_yolo_dataset(n_train: int, n_val: int, root: Path,
                       size=(640, 480), seed: int = SEED,
                       relit: bool = True) -> Path:
    """YOLO 포맷(class cx cy w h) 데이터셋 + data.yaml 생성. 정답 텍스트도 기록."""
    import random
    from .config import ENVS
    root = Path(root)
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    rows = {}
    for split, n in [("train", n_train), ("val", n_val)]:
        idir = root / "images" / split
        ldir = root / "labels" / split
        idir.mkdir(parents=True, exist_ok=True); ldir.mkdir(parents=True, exist_ok=True)
        rows[split] = []
        for i in range(n):
            text = random_plate_text(prng)
            scene, (x0, y0, x1, y1), _ = make_scene(text, rng, size=size)
            if relit and prng.random() < 0.6:
                scene = relight(scene, prng.choice(ENVS), rng)
            W, H = size
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            fn = f"{split}_{i:05d}.png"
            cv2.imwrite(str(idir / fn), scene)
            (ldir / fn.replace(".png", ".txt")).write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            rows[split].append((fn, x0, y0, x1, y1, text))
    for split in rows:
        with open(root / f"{split}_gt.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["file", "x0", "y0", "x1", "y1", "text"])
            w.writerows(rows[split])
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 1\nnames: [plate]\n", encoding="utf-8")
    return root / "data.yaml"


def train_detector(data_yaml: Path, epochs=40, imgsz=320, out=DATA / "runs",
                   seed: int = SEED) -> Path:
    """YOLOv8n 번호판 검출기 학습. best.pt 경로 반환."""
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")
    m.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=16,
            project=str(out), name="plate", exist_ok=True, verbose=False,
            plots=False, seed=seed, deterministic=True, workers=0, cache=False)
    return Path(out) / "plate" / "weights" / "best.pt"


# ---- 경량화: ONNX export + INT8 -----------------------------------------
def export_onnx(pt_path: Path, imgsz=320) -> Path:
    from ultralytics import YOLO
    p = YOLO(str(pt_path)).export(format="onnx", imgsz=imgsz, opset=13, simplify=True)
    return Path(p)


def quantize_int8(onnx_fp32: Path) -> Path:
    """ONNX 동적 INT8 양자화(가중치). onnxruntime.quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    out = Path(onnx_fp32).with_name(Path(onnx_fp32).stem + "_int8.onnx")
    quantize_dynamic(str(onnx_fp32), str(out), weight_type=QuantType.QInt8)
    return out


# ---- ONNX YOLOv8 추론(엣지 런타임) --------------------------------------
def _letterbox(img, new=320, color=114):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new, new, 3), color, np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def preprocess(img, imgsz=320):
    """YOLOv8 입력 전처리: letterbox → RGB → /255 → CHW. (x, r, dx, dy)."""
    lb, r, dx, dy = _letterbox(img, imgsz)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    return x, r, dx, dy


def quantize_static_int8(onnx_fp32: Path, calib_images, imgsz=320) -> Path:
    """정적 INT8 양자화(QDQ, per-channel) — conv까지 INT8 가속.
    calib_images: 캘리브레이션용 BGR 이미지 경로 리스트."""
    import onnx
    from onnxruntime.quantization import (quantize_static, QuantType,
                                          QuantFormat, CalibrationDataReader)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    inp_name = onnx.load(str(onnx_fp32)).graph.input[0].name

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{inp_name: preprocess(cv2.imread(str(p)), imgsz)[0]}
                            for p in calib_images])
        def get_next(self):
            return next(self.it, None)

    pre = Path(onnx_fp32).with_name(Path(onnx_fp32).stem + "_pre.onnx")
    quant_pre_process(str(onnx_fp32), str(pre))
    out = Path(onnx_fp32).with_name(Path(onnx_fp32).stem + "_int8.onnx")
    quantize_static(str(pre), str(out), _Reader(),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
    return out


class OnnxYolo:
    """onnxruntime YOLOv8 검출기(엣지). FP32/INT8 onnx 공용."""

    def __init__(self, onnx_path: Path, imgsz=320, conf=0.25, iou=0.5,
                 providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=list(providers))
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def _pre(self, img):
        return preprocess(img, self.imgsz)

    def detect(self, img):
        x, r, dx, dy = self._pre(img)
        out = self.sess.run(None, {self.inp: x})[0]          # [1, 5, 8400]
        out = np.squeeze(out, 0).T                           # [8400, 5]
        boxes_xywh, scores = out[:, :4], out[:, 4]
        keep = scores >= self.conf
        boxes_xywh, scores = boxes_xywh[keep], scores[keep]
        if len(scores) == 0:
            return []
        # xywh(letterbox px) → xyxy(원본 px)
        cx, cy, bw, bh = boxes_xywh.T
        x0 = (cx - bw / 2 - dx) / r; y0 = (cy - bh / 2 - dy) / r
        x1 = (cx + bw / 2 - dx) / r; y1 = (cy + bh / 2 - dy) / r
        xyxy = np.stack([x0, y0, x1, y1], 1)
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
            scores.tolist(), self.conf, self.iou)
        idx = np.array(idx).flatten() if len(idx) else []
        return [(xyxy[i].tolist(), float(scores[i])) for i in idx]

    def latency_ms(self, img, n=30, warmup=5):
        x, *_ = self._pre(img)
        for _ in range(warmup):
            self.sess.run(None, {self.inp: x})
        t = time.perf_counter()
        for _ in range(n):
            self.sess.run(None, {self.inp: x})
        return (time.perf_counter() - t) / n * 1000.0


# ---- OCR + 포맷 검증 ----------------------------------------------------
_HAN2DIGIT_NOISE = {}  # 필요시 교정 테이블


class PlateReader:
    """검출 + easyocr 한글 인식 + 포맷 검증."""

    def __init__(self, detector, ocr_langs=("ko", "en"), gpu=True, ocr_reader=None):
        self.det = detector                 # OnnxYolo 또는 ultralytics YOLO
        if ocr_reader is not None:           # 공유 인스턴스(메모리 절약)
            self.ocr = ocr_reader
        else:
            import easyocr
            self.ocr = easyocr.Reader(list(ocr_langs), gpu=gpu, verbose=False)

    def _detect(self, img):
        if isinstance(self.det, OnnxYolo):
            return self.det.detect(img)
        r = self.det(img, verbose=False)[0]               # ultralytics
        out = []
        for b in r.boxes:
            out.append((b.xyxy[0].tolist(), float(b.conf[0])))
        return out

    def read(self, img):
        results = []
        for (x0, y0, x1, y1), conf in self._detect(img):
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = int(x1), int(y1)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            txt = "".join(self.ocr.readtext(crop, detail=0)).replace(" ", "")
            results.append(dict(bbox=(x0, y0, x1, y1), conf=conf, text=txt,
                                valid=bool(re.match(PLATE_RE, txt))))
        return results


def char_accuracy(pred: str, gt: str) -> float:
    """문자 단위 정확도(길이 정규화 Levenshtein 보수)."""
    if not gt:
        return 0.0
    m, n = len(pred), len(gt)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (pred[i - 1] != gt[j - 1]))
            prev = cur
    return max(0.0, 1 - dp[n] / n)
