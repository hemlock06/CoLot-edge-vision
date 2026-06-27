"""#4 데모 — 번호판(①) + 점유(센서) → 분단위 정산 영수증.
일부는 실제 OCR로 번호판을 읽어 e2e(이미지→정산)를 보인다.
"""

from __future__ import annotations
import sys, random
from pathlib import Path

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

    # (라벨, 점유분, 등록 차량 여부 — 미등록이면 무단)
    scen = [
        ("정상 단시간", 7, True),  # 7분 주차 — 사용시간 과금의 가격경쟁력
        ("정상 장시간", 95, True),
        ("정상 중시간", 40, True),
        ("무단주차", 50, False),  # 미등록 차량
    ]
    print(f"{'상황':10s} {'영수증 (번호판=실제 OCR)':70s}")
    print("-" * 96)
    total = 0
    for label, dur, registered in scen:
        text = random_plate_text(prng)
        plate = correct_plate(
            "".join(ocr.readtext(render_plate(text), detail=0, allowlist=PLATE_ALLOW))
        )[0]
        e = Event(
            spot=int(prng.randint(1, 30)),
            start=0.0,
            end=float(dur),
            registered=registered,
            flicker=0,
            label="-",
        )
        r = settle(e, plate)
        total += r.amount
        ocrmark = "✓" if plate == text else f"✗(GT {text})"
        print(f"{label:10s} {r.line()}  OCR{ocrmark}")
    print("-" * 96)
    print(
        f"분당요금 {RATE}원 · 주차 1분 뒤 사용시작 · 경쟁사 30분 블록 대비 사용시간 과금으로 가격경쟁력 확보"
    )
    print(f"합계 청구 {total:,}원")


if __name__ == "__main__":
    main()
