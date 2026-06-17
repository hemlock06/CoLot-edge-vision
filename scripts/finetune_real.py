"""실데이터 fine-tune — 합성 사전학습 검출기를 실 번호판으로 적응(전이학습).
측정: (a) 합성 검출기 그대로 실 val (도메인 갭) → (b) 실 fine-tune 후 실 val.
산출: data/real_finetune.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.config import DATA, SEED
from ultralytics import YOLO

REAL = str(DATA / "real_ds" / "data.yaml")
SYN = str(DATA / "runs" / "plate" / "weights" / "best.pt")


def m50(metrics):
    try: return float(metrics.box.map50)
    except Exception: return None


def main():
    # (a) 합성 검출기 → 실 val (도메인 갭, 학습 전)
    base = YOLO(SYN).val(data=REAL, imgsz=320, verbose=False, plots=False,
                         project=str(DATA / "runs"), name="real_base", exist_ok=True)
    base_map = m50(base)

    # (b) 합성 사전학습 → 실 fine-tune (전이학습)
    m = YOLO(SYN)
    m.train(data=REAL, epochs=40, imgsz=320, batch=16, project=str(DATA / "runs"),
            name="plate_real", exist_ok=True, verbose=False, plots=False,
            workers=0, cache=False, seed=SEED, deterministic=True)
    ft = YOLO(str(DATA / "runs" / "plate_real" / "weights" / "best.pt")).val(
        data=REAL, imgsz=320, verbose=False, plots=False,
        project=str(DATA / "runs"), name="real_ft", exist_ok=True)
    ft_map = m50(ft)

    # (c) 대조: 실 데이터로 scratch 학습(사전학습 효과 비교)
    sc = YOLO("yolov8n.pt")
    sc.train(data=REAL, epochs=40, imgsz=320, batch=16, project=str(DATA / "runs"),
             name="plate_real_scratch", exist_ok=True, verbose=False, plots=False,
             workers=0, cache=False, seed=SEED, deterministic=True)
    scm = YOLO(str(DATA / "runs" / "plate_real_scratch" / "weights" / "best.pt")).val(
        data=REAL, imgsz=320, verbose=False, plots=False,
        project=str(DATA / "runs"), name="real_sc", exist_ok=True)
    sc_map = m50(scm)

    res = dict(synthetic_on_real_map50=base_map,
               finetuned_map50=ft_map, scratch_map50=sc_map)
    (DATA / "real_finetune.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print("=== 실데이터 검출 (mAP50) ===")
    print(f"  합성 검출기 그대로(도메인 갭) : {base_map:.3f}")
    print(f"  합성→실 fine-tune(전이)      : {ft_map:.3f}")
    print(f"  실 scratch(사전학습 없음)     : {sc_map:.3f}")


if __name__ == "__main__":
    main()
