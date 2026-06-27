"""#4 시간 정산 e2e — AI를 제품/사업에 연결.

코랏의 PM 차별점은 경쟁사의 *30분 블록 과금* 대비 **사용시간 분단위 정산**(가격경쟁력).
회원·번호판 등록 차량은 예약·선결제 없이, 주차 grace분 뒤 자동 사용시작 →
출차 settle초 뒤 사용시간만큼 자동 결제된다.
입차→점유(센서)→번호판(①)→출차→사용시간 요금 산출까지 한 흐름으로 연결한다.
  - 정상(등록) : (점유분 − 사용시작 유예) × 분당요금, 자동 결제
  - 무단(미등록): 미정산 + 과태료 플래그(②/스트리밍 연계)
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from .config import AnomalyCfg
from .occupancy import Event

RATE = 40  # 원/분 (KU 사업계획서: 30분 1,200원 → 40원/분)
BLOCK_MIN, BLOCK_FEE = 30, 1200  # 경쟁사 30분 블록 과금
PENALTY = 40000  # 미등록 무단주차 과태료(예시)


@dataclass
class Receipt:
    spot: int
    plate: str
    minutes: int  # 점유 분(올림)
    permin_fee: int  # 분단위 정산액
    block_fee: int  # 30분 블록 환산(경쟁사)
    saving: int  # 블록 대비 절감
    surcharge: int  # 초과 할증
    penalty: int  # 무단 과태료
    amount: int  # 최종 청구
    status: str

    def line(self) -> str:
        return (
            f"[면{self.spot:02d}] {self.plate:10s} {self.minutes:3d}분 "
            f"분단위 {self.permin_fee:>6,}원 (블록 {self.block_fee:>6,}원, "
            f"절감 {self.saving:>5,}원) {self.status}  청구 {self.amount:>7,}원"
        )


def settle(
    e: Event, plate: str, rate: int = RATE, grace: float = AnomalyCfg().start_grace_min
) -> Receipt:
    dur = max(0.0, e.end - e.start)
    minutes = math.ceil(dur)
    block_fee = math.ceil(dur / BLOCK_MIN) * BLOCK_FEE
    plate = plate or "미인식"

    if not e.registered:  # 미등록 무단점유
        return Receipt(
            e.spot,
            plate,
            minutes,
            0,
            block_fee,
            0,
            0,
            PENALTY,
            PENALTY,
            "무단(미등록·과태료)",
        )

    billable = max(0.0, dur - grace)  # 사용시작 = 주차 grace분 뒤
    base = math.ceil(billable) * rate
    return Receipt(
        e.spot,
        plate,
        minutes,
        base,
        block_fee,
        max(0, block_fee - base),
        0,
        0,
        base,
        "정산완료",
    )
