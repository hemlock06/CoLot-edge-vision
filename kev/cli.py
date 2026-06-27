"""kev CLI — 단일 이미지/데모 실행.

  python -m kev.cli demo
  python -m kev.cli adaptive <image>
  python -m kev.cli plate <image>
"""
from __future__ import annotations
import argparse
import sys
import cv2
import joblib

from .config import DATA
from .adaptive import AdaptiveSensor


def cmd_adaptive(args):
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"이미지를 못 읽음: {args.image}")
    clf_path = DATA / "adaptive_clf.joblib"
    clf = joblib.load(clf_path) if clf_path.exists() else None
    d = AdaptiveSensor(classifier=clf).step(img, motion=args.motion)
    print(f"환경={d.env}  모드={d.mode}  판독성={d.readable:.2f}  "
          f"OCR={d.run_ocr}  전력={d.power:.2f}")


def cmd_plate(args):
    from .demo import default_plate_reader
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"이미지를 못 읽음: {args.image}")
    for r in default_plate_reader(gpu=False).read(img):
        print(f"bbox={r['bbox']} conf={r['conf']:.2f} text={r['text']} valid={r['valid']}")


def cmd_demo(args):
    from .demo import run_demo
    run_demo(save_fig=not args.no_fig)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kev", description="CoLot Edge Vision")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("adaptive")
    a.add_argument("image")
    a.add_argument("--motion", type=float, default=0.3)
    a.set_defaults(fn=cmd_adaptive)

    p = sub.add_parser("plate")
    p.add_argument("image")
    p.set_defaults(fn=cmd_plate)

    d = sub.add_parser("demo")
    d.add_argument("--no-fig", action="store_true")
    d.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
