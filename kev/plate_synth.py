"""합성 데이터 생성: 한글 번호판 · 장면 합성 · 조명(환경) 증강.

원 코랏 데이터가 비공개이므로 동일 구조를 합성으로 재현한다.
  - render_plate : 한국 번호판 이미지 (맑은 고딕, 유효 포맷)
  - make_scene   : 노면 배경에 번호판을 원근 합성 (검출/OCR 입력)
  - relight      : 휘도 환경(day/low_light/glare/backlit/overexposed) 증강
조명 증강은 '적용한 변환 = 환경 정답'이므로 ③ 분류의 정직한 라벨이 된다.
"""

from __future__ import annotations
import csv
import os
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .config import ASSETS, PLATE_HANGUL, PlateCfg, SEED

# 한글 폰트 직접 지정용 환경변수(최우선). 비-Windows/CI/도커 이식성(P0-3).
FONT_ENV = "KEV_FONT"
# 리포지에 폰트를 동봉하려면 여기에 .ttf/.ttc 를 둔다(라이선스는 사용자 책임).
_BUNDLED_FONT_DIR = ASSETS / "fonts"

# OS 표준 한글 폰트 경로(우선순위 순). 첫 존재 폰트를 사용.
_FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\HMKMRHD.ttf",
    r"C:\Windows\Fonts\gulim.ttc",
    # Linux (Nanum / Noto CJK) — 배포판·도커 공통 경로
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    # macOS
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
]


def _font_candidates() -> list[str]:
    """폰트 후보 경로(우선순위 순): KEV_FONT 환경변수 → 동봉(assets/fonts) → OS 표준."""
    cands: list[str] = []
    env = os.environ.get(FONT_ENV)
    if env:
        cands.append(env)
    if _BUNDLED_FONT_DIR.is_dir():
        bundled: list[str] = []
        for ext in ("*.ttf", "*.ttc", "*.otf"):
            bundled.extend(str(p) for p in _BUNDLED_FONT_DIR.glob(ext))
        cands.extend(sorted(bundled))
    cands.extend(_FONT_CANDIDATES)
    return cands


def _resolve_font_path() -> str | None:
    """존재하는 첫 폰트 경로(없으면 None)."""
    for p in _font_candidates():
        if p and Path(p).exists():
            return p
    return None


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = _resolve_font_path()
    if path is None:
        searched = "\n  ".join(_font_candidates())
        raise FileNotFoundError(
            f"한글 폰트를 찾지 못함. 환경변수 {FONT_ENV}=<폰트.ttf> 로 지정하거나 "
            f"{_BUNDLED_FONT_DIR} 에 .ttf/.ttc 를 두세요.\n  탐색 경로:\n  {searched}"
        )
    return ImageFont.truetype(path, size)


def random_plate_text(rng: random.Random) -> str:
    """유효한 한국 번호판 문자열. 신형(3자리)·구형(2자리) 혼합."""
    nd = rng.choice([2, 3])
    head = "".join(str(rng.randint(0, 9)) for _ in range(nd))
    han = rng.choice(PLATE_HANGUL)
    tail = "".join(str(rng.randint(0, 9)) for _ in range(4))
    return f"{head}{han}{tail}"


def render_plate(text: str, cfg: PlateCfg = PlateCfg()) -> np.ndarray:
    """번호판 BGR 이미지 (흰 바탕·검은 글자·테두리)."""
    W, H = cfg.img_w, cfg.img_h
    img = Image.new("RGB", (W, H), (242, 244, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, W - 4, H - 4], outline=(20, 20, 20), width=4)
    f = _font(int(H * 0.62))
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(
        ((W - tw) / 2 - bb[0], (H - th) / 2 - bb[1]), text, font=f, fill=(15, 15, 15)
    )
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _asphalt(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """절차적 노면 배경."""
    base = rng.integers(95, 130)
    noise = rng.normal(0, 14, (h, w, 1)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = np.repeat(img, 3, axis=2)
    # 차선/얼룩 약간
    for _ in range(rng.integers(0, 3)):
        y = int(rng.integers(0, h))
        cv2.line(
            img,
            (0, y),
            (w, y),
            (int(rng.integers(140, 180)),) * 3,
            int(rng.integers(2, 6)),
        )
    return img


def make_scene(
    text: str, rng: np.random.Generator, size=(640, 480), cfg: PlateCfg = PlateCfg()
):
    """노면 배경에 번호판을 원근 합성. (scene_bgr, bbox_xyxy, plate_text) 반환."""
    W, H = size
    scene = _asphalt(W, H, rng)
    plate = render_plate(text, cfg)
    ph, pw = plate.shape[:2]

    scale = float(rng.uniform(0.30, 0.62)) * W / pw
    pw2, ph2 = int(pw * scale), int(ph * scale)
    plate = cv2.resize(plate, (pw2, ph2))

    # 약한 원근 워프
    jitter = lambda m: rng.uniform(-m, m)
    src = np.float32([[0, 0], [pw2, 0], [pw2, ph2], [0, ph2]])
    dst = src + np.float32([[jitter(pw2 * 0.06), jitter(ph2 * 0.12)] for _ in range(4)])
    Mwarp = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(plate, Mwarp, (pw2, ph2), borderValue=(110, 110, 110))

    x0 = int(rng.integers(10, max(11, W - pw2 - 10)))
    y0 = int(rng.integers(10, max(11, H - ph2 - 10)))
    roi = scene[y0 : y0 + ph2, x0 : x0 + pw2]
    mask = (warp.sum(2) > 30)[..., None]
    scene[y0 : y0 + ph2, x0 : x0 + pw2] = np.where(mask, warp, roi)
    bbox = (x0, y0, x0 + pw2, y0 + ph2)
    return scene, bbox, text


# ---- 조명(환경) 증강 -----------------------------------------------------
def relight(bgr: np.ndarray, env: str, rng: np.random.Generator) -> np.ndarray:
    """환경 라벨에 맞는 휘도 변환. 반환 이미지의 정답 환경 = env."""
    img = bgr.astype(np.float32)
    h, w = img.shape[:2]
    if env == "day_normal":
        img = img * rng.uniform(0.92, 1.08) + rng.uniform(-6, 6)
    elif env == "low_light":
        img = img * rng.uniform(0.18, 0.34)
        img += rng.normal(0, 8, img.shape)
    elif env == "overexposed":
        img = img * rng.uniform(1.5, 2.1) + rng.uniform(40, 80)
    elif env == "glare":
        cx, cy = int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h)
        Y, X = np.ogrid[:h, :w]
        rad = rng.uniform(0.12, 0.22) * (w + h)
        blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * rad**2))
        img += blob[..., None] * rng.uniform(230, 300)
    elif env == "backlit":
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        vig = d / d.max()  # 중앙 어둡고 주변 밝게
        img = img * (0.45 + 0.05) + vig[..., None] * rng.uniform(120, 180)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---- 악천후 증강 (비·안개·눈) ------------------------------------------
def add_weather(bgr: np.ndarray, weather: str, rng: np.random.Generator) -> np.ndarray:
    """악천후 열화. 반환 이미지의 정답 환경 = weather."""
    img = bgr.astype(np.float32)
    h, w = img.shape[:2]
    if weather == "rain":
        # 채도·대비·밝기 저하
        gray = img.mean(2, keepdims=True)
        img = img * 0.78 + gray * 0.10 + 14  # 탈색 + 약간 어둡게
        # 빗줄기(대각 모션블러된 밝은 선)
        layer = np.zeros((h, w), np.float32)
        ang = rng.uniform(-20, 20)
        for _ in range(int(rng.uniform(420, 700))):
            x, y = rng.integers(0, w), rng.integers(0, h)
            ln = int(rng.uniform(8, 22))
            dx = int(ln * np.sin(np.deg2rad(ang)))
            dy = int(ln * np.cos(np.deg2rad(ang)))
            cv2.line(layer, (x, y), (x + dx, y + dy), float(rng.uniform(120, 200)), 1)
        k = max(3, int(rng.uniform(7, 13)))
        layer = cv2.GaussianBlur(layer, (1, k | 1), 0)
        img += layer[..., None] * 0.7
        # 렌즈 물방울(국소 블러 패치)
        blurred = cv2.GaussianBlur(img, (0, 0), 4)
        for _ in range(int(rng.uniform(6, 14))):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            r = int(rng.uniform(6, 16))
            mask = np.zeros((h, w), np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            m = (mask > 0)[..., None]
            img = np.where(m, blurred, img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif weather == "fog":
        # 대기산란 모델: out = img*t + A*(1-t)  (airlight A, 투과율 t)
        t = float(rng.uniform(0.38, 0.6))
        A = float(rng.uniform(190, 225))
        img = img * t + A * (1 - t)
        img = img * 0.92 + img.mean(2, keepdims=True) * 0.08  # 탈색
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif weather == "snow":
        img = img * 0.95 + 16
        # 눈송이(밝은 점 + 일부 모션블러)
        flakes = np.zeros((h, w), np.float32)
        for _ in range(int(rng.uniform(260, 520))):
            x, y = rng.integers(0, w), rng.integers(0, h)
            r = int(rng.uniform(1, 4))
            cv2.circle(flakes, (x, y), r, float(rng.uniform(200, 255)), -1)
        flakes = cv2.GaussianBlur(flakes, (3, 3), 0)
        img += flakes[..., None] * 0.85
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---- variant B: OOD 검증용 2차 증강기 (같은 환경 의미, 다른 파라미터) ----
# 목적: genA로 학습한 환경분류기를 genB로 테스트 → 'A의 변환을 외운 게 아니라
# 환경 통계(휘도·대비·dark-channel)를 학습했다'는 순환성 반박 증거.
def relight_b(bgr: np.ndarray, env: str, rng: np.random.Generator) -> np.ndarray:
    """relight의 2차 생성기(variant B) — 같은 환경, 다른 파라미터(OOD 검증용)."""
    img = bgr.astype(np.float32)
    h, w = img.shape[:2]
    if env == "day_normal":
        img = img * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)
    elif env == "low_light":
        img = img * rng.uniform(0.22, 0.40) + rng.normal(0, 12, img.shape)
    elif env == "overexposed":
        img = img * rng.uniform(1.4, 1.95) + rng.uniform(55, 100)
    elif env == "glare":
        cx = int(rng.uniform(0.15, 0.85) * w)
        cy = int(rng.uniform(0.15, 0.85) * h)
        Y, X = np.ogrid[:h, :w]
        rad = rng.uniform(0.10, 0.19) * (w + h)
        blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * rad**2))
        img += blob[..., None] * rng.uniform(250, 330)
    elif env == "backlit":
        Y, X = np.ogrid[:h, :w]
        d = np.sqrt((X - w / 2) ** 2 + (Y - h / 2) ** 2)
        img = img * 0.42 + (d / d.max())[..., None] * rng.uniform(105, 165)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_weather_b(
    bgr: np.ndarray, weather: str, rng: np.random.Generator
) -> np.ndarray:
    """add_weather의 2차 생성기(variant B) — 같은 악천후, 다른 파라미터(OOD 검증용)."""
    img = bgr.astype(np.float32)
    h, w = img.shape[:2]
    if weather == "rain":
        gray = img.mean(2, keepdims=True)
        img = img * 0.74 + gray * 0.12 + 10
        layer = np.zeros((h, w), np.float32)
        ang = rng.uniform(-28, 12)
        for _ in range(int(rng.uniform(300, 560))):
            x, y = rng.integers(0, w), rng.integers(0, h)
            ln = int(rng.uniform(10, 26))
            dx = int(ln * np.sin(np.deg2rad(ang)))
            dy = int(ln * np.cos(np.deg2rad(ang)))
            cv2.line(layer, (x, y), (x + dx, y + dy), float(rng.uniform(110, 210)), 1)
        layer = cv2.GaussianBlur(layer, (1, max(3, int(rng.uniform(9, 15))) | 1), 0)
        img += layer[..., None] * 0.8
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif weather == "fog":
        t = float(rng.uniform(0.30, 0.52))
        A = float(rng.uniform(180, 218))
        img = img * t + A * (1 - t)
        img = img * 0.90 + img.mean(2, keepdims=True) * 0.10
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif weather == "snow":
        img = img * 0.93 + 20
        flakes = np.zeros((h, w), np.float32)
        for _ in range(int(rng.uniform(200, 430))):
            x, y = rng.integers(0, w), rng.integers(0, h)
            cv2.circle(
                flakes, (x, y), int(rng.uniform(1, 5)), float(rng.uniform(190, 255)), -1
            )
        flakes = cv2.GaussianBlur(flakes, (5, 5), 0)
        img += flakes[..., None] * 0.9
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_env(
    scene: np.ndarray, env: str, rng: np.random.Generator, variant: str = "A"
) -> np.ndarray:
    """환경 라벨 → 밝기/악천후 증강 디스패치. variant='B'는 OOD 검증용 2차 생성기."""
    from .config import WEATHER_ENVS

    _relight = relight_b if variant == "B" else relight
    _weather = add_weather_b if variant == "B" else add_weather
    if env in WEATHER_ENVS:
        return _weather(_relight(scene, "day_normal", rng), env, rng)
    return _relight(scene, env, rng)


def build_plate_dataset(
    n: int, out_dir: Path, seed: int = SEED, cfg: PlateCfg = PlateCfg()
):
    """검출+OCR용 합성 장면 n개 생성. labels.csv(파일,bbox,text) 기록."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    rows = []
    for i in range(n):
        text = random_plate_text(prng)
        scene, bbox, _ = make_scene(text, rng)
        fn = f"plate_{i:05d}.png"
        cv2.imwrite(str(out_dir / "images" / fn), scene)
        rows.append((fn, *bbox, text))
    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "x0", "y0", "x1", "y1", "text"])
        writer.writerows(rows)
    return out_dir / "labels.csv"
