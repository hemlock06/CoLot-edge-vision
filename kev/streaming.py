"""#2 실시간 조기 경보 (스트리밍 이상탐지).

배치(②)는 출차 후 세션 단위로 판정한다. 그러나 '불법주차 경고'의 실제 가치는
*차량이 무단·초과 주차 중일 때* 경보하는 것. 본 모니터는 분 단위 점유 스트림에서
위반이 발생하는 순간 경보를 내고, 탐지 지연·선행시간(차량이 떠나기까지 남은 시간)을
측정한다.

조기 탐지 가능: 무단점유(미예약) · 초과주차(결제초과) · stuck 고장(과도한 장시간).
세션 종료 시에만 판정 가능: ghost(초단시간)·flicker — 배치(②)가 담당(상보적).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from .config import AnomalyCfg
from .occupancy import Event


@dataclass
class Alert:
    spot: int
    time: float          # 경보 발생 시각(분)
    kind: str            # unauthorized / overstay / sensor_fault
    onset: float         # 위반 시작 시각
    delay: float         # 탐지 지연 = time - onset
    lead: float          # 선행시간 = 출차 - 경보 (차량이 아직 있는 동안의 여유)
    event_idx: int


class StreamingMonitor:
    """분 단위 점유 스트림에서 위반 발생 즉시 경보."""

    def __init__(self, cfg: AnomalyCfg = AnomalyCfg(), unauth_grace: float = 3.0,
                 stuck_min: float = 18 * 60):
        self.cfg = cfg
        self.unauth_grace = unauth_grace      # 미예약 점유 허용 지연(예약 등록 랙)
        self.stuck_min = stuck_min

    def alert_for(self, e: Event, idx: int) -> Optional[Alert]:
        """차량 점유 중 발생하는 가장 이른 경보(없으면 None)."""
        cands = []
        if not e.reserved:                                  # 무단점유
            at = e.start + self.unauth_grace
            if at < e.end:
                cands.append(("unauthorized", at, e.start))
        if e.reserved and e.paid_end > 0:                   # 초과주차
            at = e.paid_end + self.cfg.overstay_grace_min
            if at < e.end:
                cands.append(("overstay", at, e.paid_end))
        at = e.start + self.stuck_min                       # stuck 고장
        if at < e.end:
            cands.append(("sensor_fault", at, e.start))
        if not cands:
            return None
        kind, at, onset = min(cands, key=lambda c: c[1])    # 가장 이른 위반
        return Alert(e.spot, at, kind, onset, at - onset, e.end - at, idx)

    def run(self, events: List[Event]) -> List[Alert]:
        """이벤트-구동 스트림 처리 — 후보 위반시각을 시간순으로 흘리며,
        점유가 활성인 동안 위반이 처음 성립하는 시점에 경보(점유당 1회)."""
        cands = []
        for i, e in enumerate(events):
            if not e.reserved:
                cands.append((e.start + self.unauth_grace, i, "unauthorized", e.start))
            if e.reserved and e.paid_end > 0:
                cands.append((e.paid_end + self.cfg.overstay_grace_min, i, "overstay", e.paid_end))
            cands.append((e.start + self.stuck_min, i, "sensor_fault", e.start))
        cands.sort(key=lambda c: c[0])               # 시간순 스트림
        alerted, out = set(), []
        for at, i, kind, onset in cands:
            if i in alerted:
                continue
            e = events[i]
            if e.start <= at < e.end:                 # 점유 활성 중에만 경보
                out.append(Alert(e.spot, at, kind, onset, at - onset, e.end - at, i))
                alerted.add(i)
        return out
