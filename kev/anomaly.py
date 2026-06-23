"""② 불법주차 이상탐지 — 회원 원장 대조(룰) + IsolationForest(센서 이상).

설계:
  룰  : 회원 등록 원장과 대조 → unauthorized(미등록 무단) 결정(결정론적)
        + 센서 물리 위반(ghost/flicker/stuck) → sensor_fault
  ML  : 정상 세션으로 IsolationForest 학습 → 점유 패턴 이상점수
        룰이 못 잡는 subtle 센서 고장을 통계적으로 포착
최종 : 룰 우선, 미검출분은 ML 이상점수로 보강 → 이진 이상(정상/이상) + 유형
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.ensemble import IsolationForest

from .config import AnomalyCfg
from .occupancy import Event


def features(e: Event) -> np.ndarray:
    """세션 → 이상탐지 피처 (센서 점유 패턴 — 선결제 의존 피처 없음)."""
    dur = e.end - e.start
    hour = (e.start % (24 * 60)) / 60.0
    return np.array([dur, np.log1p(dur), hour, e.flicker,
                     float(e.registered)], np.float32)


FEATS = ["dur", "log_dur", "hour", "flicker", "registered"]


@dataclass
class Flag:
    idx: int
    rule: Optional[str]     # unauthorized / sensor_fault / None
    ml_anomaly: bool
    score: float
    pred: str               # normal / unauthorized / fault / anomaly


class ParkingAnomalyDetector:
    def __init__(self, cfg: AnomalyCfg = AnomalyCfg(), seed: int = 20231016):
        self.cfg = cfg
        self.iforest = IsolationForest(contamination=cfg.contamination,
                                       n_estimators=200, random_state=seed)
        self._fitted = False

    # ---- 룰: 회원 원장 대조 + 센서 물리 위반 ----
    def rule(self, e: Event) -> Optional[str]:
        dur = e.end - e.start
        if dur < self.cfg.min_occupancy_min:
            return "sensor_fault"                 # ghost(초단시간)
        if e.flicker >= self.cfg.flicker_thresh:
            return "sensor_fault"                 # flicker
        if dur > self.cfg.stuck_min:
            return "sensor_fault"                 # stuck-on
        if not e.registered:
            return "unauthorized"                 # 미등록 차량 점유(무단)
        return None

    def fit(self, normal_events: List[Event]):
        X = np.array([features(e) for e in normal_events], np.float32)
        self.iforest.fit(X)
        self._fitted = True
        return self

    def predict(self, events: List[Event]) -> List[Flag]:
        X = np.array([features(e) for e in events], np.float32)
        scores = self.iforest.score_samples(X) if self._fitted else np.zeros(len(events))
        ml_anom = self.iforest.predict(X) == -1 if self._fitted else np.zeros(len(events), bool)
        out = []
        for i, e in enumerate(events):
            r = self.rule(e)
            if r is not None:
                pred = "fault" if r == "sensor_fault" else r
            elif ml_anom[i]:
                pred = "anomaly"                  # 룰 미검출 + 통계 이상
            else:
                pred = "normal"
            out.append(Flag(i, r, bool(ml_anom[i]), float(scores[i]), pred))
        return out
