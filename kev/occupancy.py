"""② 주차 점유 시뮬레이터 — 점유 이벤트 + 회원 등록 원장 생성.

코랏 카스토퍼는 주차면 점유(센서) + 앱 회원 원장(등록 번호판)을 함께 가진다.
정산은 예약/선결제가 아니라, 등록 차량이 주차하면 자동으로 사용시작·시간과금된다.
이상 유형:
  normal       : 회원 등록 차량의 정상 주차 (자동 사용시작·시간과금)
  unauthorized : 미등록 차량의 점유 (= 불법·무단주차, '불법주차 경고' 기능)
  fault        : 센서 고장 — stuck(과도한 장시간) · flicker(토글 폭주) · ghost(초단시간)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Event:
    spot: int
    start: float          # 분 (t0 기준)
    end: float
    registered: bool      # 앱 회원·번호판 등록 차량 여부
    flicker: int          # 센서 토글 수(고장 신호)
    label: str            # 정답 라벨


def simulate(n_events: int = 1200, n_spots: int = 40, days: int = 14,
             seed: int = 20231016,
             mix=(("normal", .80), ("unauthorized", .13), ("fault", .07)),
             random_faults: bool = False) -> List[Event]:
    """random_faults=True: 고장을 *명백/subtle 설계 없이* 연속 파라미터로 생성
    → 자기충족 ablation 반박용(룰·ML 모두 데이터에 맞춰지지 않은 상태에서 평가)."""
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
        registered, flicker = True, 0

        if lab == "normal":
            registered = True
        elif lab == "unauthorized":
            registered = False                              # 미등록 차량 점유
        elif lab == "fault" and random_faults:
            # 연속 파라미터: 한 축(flicker/장시간/초단시간)에 임의 크기 이상.
            # 룰 임계(2분·20·18h)를 연속으로 가로질러 명백/subtle 구분 없음.
            registered = bool(rng.random() < 0.5)
            axis = int(rng.integers(0, 3))
            if axis == 0:                                   # flicker 임의 크기
                flicker = int(rng.integers(5, 80)); dur = float(rng.uniform(10, 120))
            elif axis == 1:                                 # 장시간(연속)
                dur = float(np.exp(rng.uniform(np.log(6 * 60), np.log(72 * 60))))
            else:                                           # 초단시간(연속)
                dur = float(rng.uniform(0.3, 5.0))
        elif lab == "fault":
            ftype = rng.integers(0, 4)
            if ftype == 0:            # stuck-on(명백) — 룰 포착
                dur = float(rng.uniform(20 * 60, 72 * 60))
                registered = bool(rng.random() < .5)
            elif ftype == 1:          # flicker(명백, ≥22) — 룰 포착
                flicker = int(rng.integers(22, 90))
                dur = float(rng.uniform(5, 60))
                registered = bool(rng.random() < .5)
            elif ftype == 2:          # ghost(명백, <2분) — 룰 포착
                dur = float(rng.uniform(0.3, 1.6))
                registered = False
            else:
                # subtle 고장: 회원 등록 정상 차량인데 센서만 약하게 이상.
                # → unauthorized/명백고장 룰 모두 회피, IForest만 포착.
                registered = True
                if rng.random() < 0.5:           # 약한 flicker(8~18, 룰 임계 미만)
                    flicker = int(rng.integers(8, 18))
                    dur = float(rng.uniform(30, 120))
                else:                            # 짧은 ghost(2.5~4.5분, 룰 회피)
                    dur = float(rng.uniform(2.5, 4.5))
        end = start + dur
        evs.append(Event(spot, start, end, registered, flicker, lab))
    evs.sort(key=lambda e: e.start)
    return evs
