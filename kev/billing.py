"""#4 분단위 정산 e2e — AI를 제품/사업에 연결.

코랏의 PM 차별점은 경쟁사의 *30분 블록 과금* 대비 **분 단위 정산**(가격경쟁력).
입차→점유(센서)→번호판(①)→출차→분단위 요금 산출까지 한 흐름으로 연결한다.
  - 정상     : 점유 분 × 분당요금
  - 초과주차 : 결제분 + 초과분 할증(②/스트리밍 연계)
  - 무단점유 : 미정산 + 과태료 플래그
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from .config import AnomalyCfg
from .occupancy import Event

RATE = 40                 # 원/분 (KU 사업계획서: 30분 1,200원 → 40원/분)
BLOCK_MIN, BLOCK_FEE = 30, 1200   # 경쟁사 30분 블록 과금
SURCHARGE = 1.5           # 초과주차 할증
PENALTY = 40000           # 무단주차 과태료(예시)


@dataclass
class Receipt:
    spot: int
    plate: str
    minutes: int          # 점유 분(올림)
    permin_fee: int       # 분단위 정산액
    block_fee: int        # 30분 블록 환산(경쟁사)
    saving: int           # 블록 대비 절감
    surcharge: int        # 초과 할증
    penalty: int          # 무단 과태료
    amount: int           # 최종 청구
    status: str

    def line(self) -> str:
        return (f"[면{self.spot:02d}] {self.plate:10s} {self.minutes:3d}분 "
                f"분단위 {self.permin_fee:>6,}원 (블록 {self.block_fee:>6,}원, "
                f"절감 {self.saving:>5,}원) {self.status}  청구 {self.amount:>7,}원")


def settle(e: Event, plate: str, rate: int = RATE,
           grace: float = AnomalyCfg().overstay_grace_min) -> Receipt:
    dur = max(0.0, e.end - e.start)
    minutes = math.ceil(dur)
    block_fee = math.ceil(dur / BLOCK_MIN) * BLOCK_FEE
    plate = plate or "미인식"

    if not e.reserved:                                   # 무단점유
        return Receipt(e.spot, plate, minutes, 0, block_fee, 0, 0, PENALTY,
                       PENALTY, "무단(미정산·과태료)")

    paid_min = max(0.0, e.paid_end - e.start) if e.paid_end > 0 else dur
    over = max(0.0, dur - paid_min)
    base = math.ceil(min(dur, paid_min)) * rate
    surcharge = math.ceil(over) * int(rate * SURCHARGE) if over > grace else 0
    amount = base + surcharge
    status = "초과정산(할증)" if surcharge > 0 else "정산완료"
    return Receipt(e.spot, plate, minutes, base, block_fee,
                   max(0, block_fee - base), surcharge, 0, amount, status)
