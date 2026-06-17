"""② 주차 점유 시뮬레이터 — 점유 이벤트 + 예약/결제 원장 생성.

코랏 카스토퍼는 주차면 점유(센서) + 앱 예약/결제(원장)를 함께 가진다.
이상 유형:
  normal       : 예약·결제된 정상 주차
  unauthorized : 예약/결제 없이 점유 (= 불법·무단주차, '불법주차 경고' 기능)
  overstay     : 결제시간 초과 점유
  fault        : 센서 고장 — stuck(과도한 장시간) · flicker(토글 폭주) · ghost(초단시간)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List
import numpy as np


@dataclass
class Event:
    spot: int
    start: float          # 분 (t0 기준)
    end: float
    reserved: bool        # 앱 예약 존재
    paid_end: float       # 결제로 보장된 종료시각(분); 미예약이면 -1
    flicker: int          # 센서 토글 수(고장 신호)
    label: str            # 정답 라벨


def simulate(n_events: int = 1200, n_spots: int = 40, days: int = 14,
             seed: int = 20231016,
             mix=(("normal", .72), ("unauthorized", .12),
                  ("overstay", .10), ("fault", .06))) -> List[Event]:
    rng = np.random.default_rng(seed)
    labels = [m[0] for m in mix]
    probs = np.array([m[1] for m in mix]); probs /= probs.sum()
    horizon = days * 24 * 60
    evs: List[Event] = []
    for _ in range(n_events):
        lab = str(rng.choice(labels, p=probs))
        spot = int(rng.integers(0, n_spots))
        start = float(rng.uniform(0, horizon - 24 * 60))
        # 정상 체류시간: 로그정규(중앙 ~45분), 주간 편향
        dur = float(np.clip(rng.lognormal(3.7, 0.6), 5, 360))
        reserved, paid_end, flicker = True, 0.0, 0

        if lab == "normal":
            paid_end = start + dur * float(rng.uniform(1.0, 1.4))
        elif lab == "unauthorized":
            reserved, paid_end = False, -1.0
        elif lab == "overstay":
            paid_min = dur * float(rng.uniform(0.4, 0.8))   # 결제는 짧게
            paid_end = start + paid_min
        elif lab == "fault":
            ftype = rng.integers(0, 4)
            if ftype == 0:            # stuck-on(명백) — 룰 포착
                dur = float(rng.uniform(20 * 60, 72 * 60))
                reserved, paid_end = (rng.random() < .5), -1.0
            elif ftype == 1:          # flicker(명백, ≥22) — 룰 포착
                flicker = int(rng.integers(22, 90))
                dur = float(rng.uniform(5, 60))
                reserved, paid_end = (rng.random() < .5), -1.0
            elif ftype == 2:          # ghost(명백, <2분) — 룰 포착
                dur = float(rng.uniform(0.3, 1.6))
                reserved, paid_end = False, -1.0
            else:
                # subtle 고장: 원장은 정상(예약·결제 OK)인데 센서만 약하게 이상.
                # → unauthorized/overstay/명백고장 룰 모두 회피, IForest만 포착.
                reserved = True
                if rng.random() < 0.5:           # 약한 flicker(8~18, 룰 임계 미만)
                    flicker = int(rng.integers(8, 18))
                    dur = float(rng.uniform(30, 120))
                else:                            # 짧은 ghost(2.5~4.5분, 룰 회피)
                    dur = float(rng.uniform(2.5, 4.5))
                paid_end = start + dur * float(rng.uniform(1.15, 1.5))
        end = start + dur
        if reserved and paid_end > 0 and lab == "normal":
            pass
        evs.append(Event(spot, start, end, reserved, paid_end, flicker, lab))
    evs.sort(key=lambda e: e.start)
    return evs
