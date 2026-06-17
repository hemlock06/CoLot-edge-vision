"""공용 상수·경로·설정."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"
ASSETS = ROOT / "assets"
for _p in (DATA, FIGS, ASSETS):
    _p.mkdir(parents=True, exist_ok=True)

# ---- ③ 휘도 환경적응 ----------------------------------------------------
# 밝기 계열 환경 (휘도 분포로 식별)
LIGHT_ENVS = ["day_normal", "low_light", "glare", "backlit", "overexposed"]
# 악천후 계열 (실외 로버스트니스 확장 — 원 휘도 특허 범위 밖, 명시적 구분)
WEATHER_ENVS = ["rain", "fog", "snow"]
ENVS = LIGHT_ENVS + WEATHER_ENVS

# 센싱 모드 — 카스토퍼가 휘도 판단으로 선택하는 동작
#   rgb_full : RGB 카메라 풀 추론 (가장 비쌈)
#   rgb_boost: 노출/게인 보정 후 RGB 추론
#   ir       : 적외선 폴백 (야간/저조도)
#   skip     : 절전 — 차량 활동 없음/판독 불가
MODES = ["rgb_full", "rgb_boost", "ir", "skip"]

# 모드별 상대 센싱 전력 (rgb_full = 1.0 기준; 카스토퍼 저전력 설계 반영)
POWER_COST = {"rgb_full": 1.00, "rgb_boost": 1.15, "ir": 0.55, "skip": 0.05}
# 번호판 추론(검출+OCR) 1회 연산 비용. 엣지에선 추론이 에너지를 지배.
OCR_COST = 0.80


@dataclass
class AdaptiveCfg:
    glare_sat_hi: float = 0.18      # 과포화 화소 비율 임계 (글레어)
    dark_p95: float = 70.0          # 상위 5% 밝기가 이만큼 낮으면 저조도
    over_mean: float = 205.0        # 평균 밝기 과노출 임계
    backlit_split: float = 95.0     # 중앙-주변 밝기 격차 (역광)
    motion_skip: float = 0.012      # 활동도(모션) 이 값 미만이면 절전 후보
    readable_min: float = 0.45      # 판독가능성 추정 하한 (OCR 시도 여부)
    fog_dark: float = 140.0         # 안개 dark-channel 임계
    fog_dyn: float = 85.0           # 안개 동적범위 상한


# ---- ① 번호판 -----------------------------------------------------------
# 한국 번호판: 2~3자리 숫자 + 한글 1 + 4자리 숫자  예) 12가3456 / 123가4567
PLATE_HANGUL = list("가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주바사아자하허호배")
PLATE_RE = r"^\d{2,3}[가-힣]\d{4}$"


@dataclass
class PlateCfg:
    img_w: int = 520
    img_h: int = 110
    det_imgsz: int = 320            # 엣지 검출 입력 해상도
    ocr_langs: tuple = ("ko", "en")


# ---- ② 불법주차 이상탐지 ------------------------------------------------
@dataclass
class AnomalyCfg:
    overstay_grace_min: float = 10.0   # 결제시간 초과 허용(분)
    contamination: float = 0.03        # IsolationForest 오염도(오탐 억제)
    min_occupancy_min: float = 2.0     # 이보다 짧은 점유는 노이즈


SEED = 20231016  # 코랏 국방대제전 금상(23.09) 무렵 — 재현용 고정 시드
