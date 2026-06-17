"""#4 데모 — 번호판(①) + 점유(센서) → 분단위 정산 영수증.
일부는 실제 OCR로 번호판을 읽어 e2e(이미지→정산)를 보인다.
"""
from __future__ import annotations
import sys, random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.config import SEED
from kev.plate_synth import render_plate, random_plate_text
from kev.plate import correct_plate, PLATE_ALLOW
from kev.occupancy import Event
from kev.billing import settle, RATE


def main():
    import easyocr
    ocr = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    prng = random.Random(SEED)

    # (라벨, 점유분, 결제분 또는 None=미예약)
    scen = [
        ("정상 단시간",  7,   30),    # 7분 주차 — 분단위의 가격경쟁력
        ("정상 장시간",  95,  120),
        ("초과주차",     80,  40),     # 결제 40분, 80분 점유
        ("무단주차",     50,  None),   # 미예약
    ]
    print(f"{'상황':10s} {'영수증 (번호판=실제 OCR)':70s}")
    print("-" * 96)
    total = 0
    for label, dur, paid in scen:
        text = random_plate_text(prng)
        plate = correct_plate("".join(
            ocr.readtext(render_plate(text), detail=0, allowlist=PLATE_ALLOW)))[0]
        reserved = paid is not None
        paid_end = (0 + paid) if reserved else -1.0
        e = Event(spot=int(prng.randint(1, 30)), start=0.0, end=float(dur),
                  reserved=reserved, paid_end=float(paid_end), flicker=0,
                  label="-")
        r = settle(e, plate)
        total += r.amount
        ocrmark = "✓" if plate == text else f"✗(GT {text})"
        print(f"{label:10s} {r.line()}  OCR{ocrmark}")
    print("-" * 96)
    print(f"분당요금 {RATE}원 · 경쟁사 30분 블록 대비 분단위 정산으로 단시간 이용 가격경쟁력 확보")
    print(f"합계 청구 {total:,}원")


if __name__ == "__main__":
    main()
