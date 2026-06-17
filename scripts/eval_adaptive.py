"""③ 휘도 환경적응 평가.
  (A) 환경분류: 학습 분류기(RandomForest) vs 룰 베이스라인 — 정확도/혼동행렬
  (B) 전력-커버리지: 적응 정책 vs 상시 RGB — 추론예산 절감 & 번호판 캡처 유지
산출: figs/adaptive_confusion.png, figs/adaptive_tradeoff.png, data/adaptive_clf.joblib
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean
use_korean()
from kev.config import ENVS, POWER_COST, OCR_COST, DATA, FIGS, SEED
from kev.plate_synth import make_scene, relight, apply_env, random_plate_text, _asphalt
from kev.adaptive import (AdaptiveSensor, brightness_features, feature_vector,
                          rule_label, readability)
import random

# 환경 사전확률(현장 분포 가정: 주간 다수, 야간/역광·악천후 분산)
ENV_PRIOR = {"day_normal": .42, "low_light": .18, "backlit": .10,
             "glare": .07, "overexposed": .05,
             "rain": .10, "fog": .04, "snow": .04}
# 환경별 "번호판 본질 복원가능" 여부(상시 RGB로도 회수 가능한가)
RECOVERABLE = {"day_normal": True, "backlit": True, "low_light": True,
               "glare": False, "overexposed": False,
               "rain": True, "fog": False, "snow": True}


def _gen(n_scenes, seed):
    """장면 n_scenes개 × 환경 5종 → (X 피처, y 환경). 장면 풀은 seed로 분리."""
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    X, y = [], []
    for _ in range(n_scenes):
        scene, _, _ = make_scene(random_plate_text(prng), rng)
        for env in ENVS:
            img = apply_env(scene, env, rng)
            # 현실 노이즈(센서/광학) — 클래스 경계를 흐려 정직한 난이도 부여
            if rng.random() < 0.5:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            img = np.clip(img.astype(np.float32) + rng.normal(0, 9, img.shape),
                          0, 255).astype(np.uint8)
            X.append(feature_vector(brightness_features(img)))
            y.append(env)
    return np.array(X, np.float32), np.array(y)


def eval_classifier(seed=SEED):
    # 장면 단위 분리 — train/test가 같은 장면을 공유하지 않음(누수 방지)
    Xtr, ytr = _gen(90, seed)
    Xte, yte = _gen(40, seed + 777)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    # 룰 베이스라인 (같은 테스트셋)
    from kev.adaptive import FEATURE_ORDER
    rule_pred = [rule_label({k: v for k, v in zip(FEATURE_ORDER, row)}) for row in Xte]

    acc_clf = accuracy_score(yte, pred)
    acc_rule = accuracy_score(yte, rule_pred)
    f1_clf = f1_score(yte, pred, average="macro")
    f1_rule = f1_score(yte, rule_pred, average="macro")
    joblib.dump(clf, DATA / "adaptive_clf.joblib")

    cm = confusion_matrix(yte, pred, labels=ENVS)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(ENVS))); ax.set_yticks(range(len(ENVS)))
    ax.set_xticklabels(ENVS, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ENVS, fontsize=8)
    for i in range(len(ENVS)):
        for j in range(len(ENVS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"환경분류 혼동행렬 — RF acc={acc_clf:.2f} (rule {acc_rule:.2f})")
    fig.tight_layout(); fig.savefig(FIGS / "adaptive_confusion.png", dpi=130)
    plt.close(fig)
    return dict(acc_clf=acc_clf, acc_rule=acc_rule,
                f1_clf=f1_clf, f1_rule=f1_rule, n_test=len(yte))


def eval_tradeoff(n_frames=600, seed=SEED + 1):
    """전력-커버리지: 적응 정책 vs 상시 RGB."""
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    clf = joblib.load(DATA / "adaptive_clf.joblib")
    sensor = AdaptiveSensor(classifier=clf)

    envs = list(ENV_PRIOR); probs = np.array(list(ENV_PRIOR.values()))
    probs /= probs.sum()

    pwr_adp = pwr_base = 0.0
    ocr_adp = ocr_base = 0
    recoverable = captured_adp = 0
    base_scene = None
    for _ in range(n_frames):
        car = rng.random() < 0.45                      # 차량 존재 여부
        if car:
            env = str(rng.choice(envs, p=probs))
            scene, _, _ = make_scene(random_plate_text(prng), rng)
            img = apply_env(scene, env, rng)
            motion = float(rng.uniform(0.05, 0.5))
        else:
            env = str(rng.choice(envs, p=probs))
            img = apply_env(_asphalt(640, 480, rng), env, rng)  # 빈 노면
            motion = float(rng.uniform(0.0, 0.008))

        d = sensor.step(img, motion=motion)
        pwr_adp += d.power
        pwr_base += POWER_COST["rgb_full"] + OCR_COST  # 상시: 풀 RGB + 매 프레임 OCR
        ocr_base += 1                                  # 상시: 매 프레임 OCR 시도
        ocr_adp += int(d.run_ocr)

        if car and RECOVERABLE[env]:                   # 진짜 회수가능한 캡처
            recoverable += 1
            captured_adp += int(d.run_ocr and d.mode != "skip")

    res = dict(
        n_frames=n_frames,
        power_saved=1 - pwr_adp / pwr_base,
        ocr_calls_reduced=1 - ocr_adp / ocr_base,
        coverage_retained=captured_adp / max(1, recoverable),
        recoverable=recoverable,
    )
    # 막대 그래프
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.6))
    ax[0].bar(["상시 RGB", "적응 정책"], [pwr_base, pwr_adp],
              color=["#9aa1ac", "#2E8B4E"])
    ax[0].set_title(f"추론 전력 예산  −{res['power_saved']*100:.0f}%")
    ax[0].set_ylabel("누적 전력비용(rgb_full=1)")
    ax[1].bar(["회수가능 캡처\n유지율", "OCR 호출\n감소"],
              [res["coverage_retained"]*100, res["ocr_calls_reduced"]*100],
              color=["#2E8B4E", "#E0962A"])
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("%")
    ax[1].set_title("커버리지 유지 · 낭비 추론 제거")
    fig.tight_layout(); fig.savefig(FIGS / "adaptive_tradeoff.png", dpi=130)
    plt.close(fig)
    return res


if __name__ == "__main__":
    c = eval_classifier()
    t = eval_tradeoff()
    print("=== (A) 환경분류 ===")
    print(f"  RF   acc={c['acc_clf']:.3f}  macroF1={c['f1_clf']:.3f}")
    print(f"  rule acc={c['acc_rule']:.3f}  macroF1={c['f1_rule']:.3f}  (n_test={c['n_test']})")
    print("=== (B) 전력-커버리지 ===")
    print(f"  전력 절감          {t['power_saved']*100:5.1f}%")
    print(f"  OCR 호출 감소       {t['ocr_calls_reduced']*100:5.1f}%")
    print(f"  회수가능 캡처 유지  {t['coverage_retained']*100:5.1f}%  (recoverable={t['recoverable']})")
    (DATA / "adaptive_metrics.json").write_text(
        json.dumps({"classify": c, "tradeoff": t}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("saved figs + data/adaptive_metrics.json")
