"""③ 휘도 기반 환경적응 센싱.

국방 표적감시 이전기술의 핵심 = "휘도값에 따른 환경 식별 → 카메라 활성화 여부 결정".
프레임의 휘도(luminance) 분포에서 피처를 뽑아 환경을 분류하고,
센싱 모드(RGB/IR/절전)와 판독가능성을 결정한다.

설계 의도(코랏 카스토퍼):
  - 차량 없거나 판독 불가한 프레임에 풀 추론을 낭비하지 않음 → 저전력·내구성
  - 저조도엔 IR 폴백, 역광/글레어엔 노출 보정 → 번호판 인식 커버리지 유지
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

from .config import AdaptiveCfg, POWER_COST, OCR_COST


def luminance(bgr: np.ndarray) -> np.ndarray:
    """BGR → 휘도(Y) [0,255] float32."""
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)


def brightness_features(bgr: np.ndarray) -> dict:
    """휘도 분포 + 공간 구조 피처. 환경 식별의 입력."""
    y = luminance(bgr)
    h, w = y.shape
    flat = y.ravel()
    p05, p50, p95 = np.percentile(flat, [5, 50, 95])
    sat_hi = float(np.mean(flat >= 245.0))     # 과포화(글레어/과노출)
    sat_lo = float(np.mean(flat <= 12.0))      # 클리핑(암부)
    dyn_range = float(p95 - p05)

    # 중앙 ROI(번호판이 잡히는 영역) vs 주변 밝기 격차 → 역광 판단
    cy0, cy1 = int(h * 0.35), int(h * 0.65)
    cx0, cx1 = int(w * 0.30), int(w * 0.70)
    center = float(y[cy0:cy1, cx0:cx1].mean())
    border = float((flat.sum() - y[cy0:cy1, cx0:cx1].sum()) /
                   max(1, flat.size - (cy1 - cy0) * (cx1 - cx0)))

    # 국소 대비(라플라시안 분산) → 텍스트 가독 신호의 대용치
    lap_var = float(cv2.Laplacian(y, cv2.CV_32F, ksize=3).var())

    # 글레어 덩어리 비율(가장 큰 과포화 연결요소)
    hot = (y >= 245.0).astype(np.uint8)
    glare_blob = 0.0
    if hot.any():
        n, _, stats, _ = cv2.connectedComponentsWithStats(hot, 8)
        if n > 1:
            glare_blob = float(stats[1:, cv2.CC_STAT_AREA].max() / y.size)

    # 안개 지표: dark channel prior (맑은 실외=낮음, 안개=높음) + 색 채도
    dark_channel = float(bgr.min(axis=2).mean())
    sat_mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean())

    # 비 지표: 방향성 그래디언트 이방성 (빗줄기는 한 방향으로 정렬 → 방향 히스토그램 피크)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    omag = np.sqrt(gx * gx + gy * gy)
    ori = np.arctan2(gy, gx) % np.pi                     # 무향 선분 → [0,π)
    oh, _ = np.histogram(ori, bins=18, range=(0, np.pi), weights=omag)
    oh = oh / (oh.sum() + 1e-6)
    grad_aniso = float(oh.max() - oh.mean())             # 방향 피크(이방성) → 비

    # 눈 지표: 작은 밝은 점 밀도 (white top-hat = 원본 − 열림 → 작은 고휘도 반점)
    _ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    speckle = float(cv2.morphologyEx(y, cv2.MORPH_TOPHAT, _ker).mean())

    return dict(mean=float(flat.mean()), std=float(flat.std()),
                p05=float(p05), p50=float(p50), p95=float(p95),
                sat_hi=sat_hi, sat_lo=sat_lo, dyn_range=dyn_range,
                center=center, border=border, backlit=border - center,
                lap_var=lap_var, glare_blob=glare_blob,
                dark_channel=dark_channel, sat_mean=sat_mean,
                grad_aniso=grad_aniso, speckle=speckle)


FEATURE_ORDER = ["mean", "std", "p05", "p50", "p95", "sat_hi", "sat_lo",
                 "dyn_range", "backlit", "lap_var", "glare_blob",
                 "dark_channel", "sat_mean", "grad_aniso", "speckle"]


def feature_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)


def rule_label(feats: dict, cfg: AdaptiveCfg = AdaptiveCfg()) -> str:
    """휘도 피처 → 환경 라벨(룰). 약지도 라벨러이자 폴백.
    악천후(비/눈)는 휘도만으론 어려워 룰 베이스라인이 약함 — RF가 보강."""
    # 안개: dark-channel prior 높고 대비 낮음 (원리적 지표라 룰로도 잡힘)
    if feats["dark_channel"] >= cfg.fog_dark and feats["dyn_range"] <= cfg.fog_dyn:
        return "fog"
    if feats["glare_blob"] >= cfg.glare_sat_hi or feats["sat_hi"] >= cfg.glare_sat_hi:
        return "glare"
    if feats["mean"] >= cfg.over_mean and feats["sat_hi"] >= 0.05:
        return "overexposed"
    if feats["p95"] <= cfg.dark_p95:
        return "low_light"
    if feats["backlit"] >= cfg.backlit_split:
        return "backlit"
    return "day_normal"


def readability(feats: dict) -> float:
    """번호판 판독가능성 추정 [0,1]. 대비·동적범위 높고 포화/암부 적을수록 ↑."""
    contrast = np.clip(feats["dyn_range"] / 180.0, 0, 1)
    sharp = np.clip(feats["lap_var"] / 400.0, 0, 1)
    spoil = np.clip(feats["sat_hi"] * 2.0 + feats["sat_lo"] * 1.5 +
                    feats["glare_blob"] * 2.5, 0, 1)
    return float(np.clip(0.5 * contrast + 0.5 * sharp - 0.6 * spoil, 0, 1))


@dataclass
class Decision:
    env: str
    mode: str
    readable: float
    run_ocr: bool
    power: float


class AdaptiveSensor:
    """휘도 환경적응 정책. classifier 없으면 룰, 있으면 학습 분류기 사용."""

    def __init__(self, cfg: AdaptiveCfg = AdaptiveCfg(), classifier=None):
        self.cfg = cfg
        self.clf = classifier        # sklearn 호환 (predict)

    def classify(self, feats: dict) -> str:
        if self.clf is not None:
            return str(self.clf.predict(feature_vector(feats)[None, :])[0])
        return rule_label(feats, self.cfg)

    def policy(self, env: str, motion: float) -> str:
        """환경 × 활동도 → 센싱 모드.
        모션이 없으면(차량 부재) 센서 절전; 차량 존재 시 휘도로 모드 선택."""
        if motion < self.cfg.motion_skip:
            return "skip"                     # 차량 없음 → 듀티사이클 절전
        if env == "low_light":
            return "ir"                       # 저조도 → 적외선 폴백
        if env in ("glare", "backlit", "overexposed", "rain", "fog", "snow"):
            return "rgb_boost"                # 보정/대비복원 후 RGB
        return "rgb_full"

    def step(self, bgr: np.ndarray, motion: float = 1.0) -> Decision:
        feats = brightness_features(bgr)
        env = self.classify(feats)
        read = readability(feats)
        mode = self.policy(env, motion)
        # 추론(검출+OCR)은 센싱이 켜져 있고 판독가능할 때만 — 낭비 추론 제거.
        # IR 폴백은 저조도를 복원하므로 판독가능으로 간주.
        run_ocr = mode != "skip" and (mode == "ir" or read >= self.cfg.readable_min)
        power = POWER_COST[mode] + (OCR_COST if run_ocr else 0.0)
        return Decision(env=env, mode=mode, readable=read,
                        run_ocr=run_ocr, power=power)
