"""통합 엣지 파이프라인 — ③ 환경적응 → ① 번호판 → ② 이상탐지.

카스토퍼 한 대의 1회 입출차 처리 흐름:
  프레임 도착
    → ③ 휘도 환경판단 → 센싱 모드(RGB/IR/절전) · 추론 여부
       → (추론 시) ① 검출+OCR → 번호판
    → 출차 시 점유 세션 + 예약/결제 원장 → ② 이상 판정(무단/초과/고장)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import joblib

from .config import DATA, AnomalyCfg
from .adaptive import AdaptiveSensor, Decision
from .anomaly import ParkingAnomalyDetector
from .occupancy import Event, simulate


@dataclass
class Record:
    env: str
    mode: str
    run_ocr: bool
    power: float
    plate: Optional[str]
    plate_valid: bool
    anomaly: Optional[str]      # 출차 시 채워짐


class CoLotEdgePipeline:
    def __init__(self, plate_reader, adaptive_clf=None, anomaly_det=None):
        self.adaptive = AdaptiveSensor(classifier=adaptive_clf)
        self.plate = plate_reader
        self.anom = anomaly_det

    def on_frame(self, frame: np.ndarray, motion: float) -> Record:
        d: Decision = self.adaptive.step(frame, motion)
        plate, valid = None, False
        if d.run_ocr and self.plate is not None:
            reads = self.plate.read(frame)
            if reads:
                best = max(reads, key=lambda r: r["conf"])
                plate, valid = best["text"], best["valid"]
        return Record(env=d.env, mode=d.mode, run_ocr=d.run_ocr, power=d.power,
                      plate=plate, plate_valid=valid, anomaly=None)

    def on_exit(self, rec: Record, event: Event) -> Record:
        if self.anom is not None:
            rec.anomaly = self.anom.predict([event])[0].pred
        return rec


def build_pipeline(plate_reader, gpu=False) -> CoLotEdgePipeline:
    """학습된 산출물 로드 + 이상탐지기 정상데이터 적합."""
    clf = None
    p = DATA / "adaptive_clf.joblib"
    if p.exists():
        clf = joblib.load(p)
    normals = [e for e in simulate(n_events=1000) if e.label == "normal"]
    anom = ParkingAnomalyDetector(AnomalyCfg()).fit(normals)
    return CoLotEdgePipeline(plate_reader, adaptive_clf=clf, anomaly_det=anom)
