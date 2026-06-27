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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean
use_korean()
from kev.config import ENVS, POWER_COST, OCR_COST, DATA, FIGS, SEED
from kev.plate_synth import make_scene, apply_env, random_plate_text, _asphalt
from kev.adaptive import (AdaptiveSensor, brightness_features, feature_vector,
                          rule_label)
import random

# 환경 사전확률(현장 분포 가정: 주간 다수, 야간/역광·악천후 분산)
ENV_PRIOR = {"day_normal": .42, "low_light": .18, "backlit": .10,
             "glare": .07, "overexposed": .05,
             "rain": .10, "fog": .04, "snow": .04}
# 환경별 "번호판 본질 복원가능" 여부(상시 RGB로도 회수 가능한가)
RECOVERABLE = {"day_normal": True, "backlit": True, "low_light": True,
               "glare": False, "overexposed": False,
               "rain": True, "fog": False, "snow": True}


def _gen(n_scenes, seed, variant="A"):
    """장면 n_scenes개 × 환경 8종 → (X 피처, y 환경). variant='B'=OOD 2차 생성기."""
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    X, y = [], []
    for _ in range(n_scenes):
        scene, _, _ = make_scene(random_plate_text(prng), rng)
        for env in ENVS:
            img = apply_env(scene, env, rng, variant=variant)
            # 현실 노이즈(센서/광학) — 클래스 경계를 흐려 정직한 난이도 부여
            if rng.random() < 0.5:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            img = np.clip(img.astype(np.float32) + rng.normal(0, 9, img.shape),
                          0, 255).astype(np.uint8)
            X.append(feature_vector(brightness_features(img)))
            y.append(env)
    return np.array(X, np.float32), np.array(y)


def eval_classifier(seed=SEED):
    # 장면 단위 분리(누수 방지) — train/test 장면 풀이 다름
    Xtr, ytr = _gen(90, seed, "A")
    Xte, yte = _gen(40, seed + 777, "A")        # in-distribution 테스트
    # ★ OOD: 다른 파라미터의 2차 생성기(B)로 테스트 — 순환성 반박
    Xood, yood = _gen(40, seed + 777, "B")
    clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc_ood = accuracy_score(yood, clf.predict(Xood))
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
    return dict(acc_clf=acc_clf, acc_rule=acc_rule, acc_ood=acc_ood,
                f1_clf=f1_clf, f1_rule=f1_rule, n_test=len(yte))


def eval_tradeoff(n_frames=600, seed=SEED + 1):
    """추론 예산 절감 vs 캡처 유지.
    ★ 1차 지표 = '추론 호출 감소'(셀 수 있는 실측). 전력은 가정 비용에 의존하므로
    민감도 스윕으로 범위(min~max)를 함께 보고(헤드라인 단정 회피)."""
    from collections import Counter
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    clf = joblib.load(DATA / "adaptive_clf.joblib")
    sensor = AdaptiveSensor(classifier=clf)
    envs = list(ENV_PRIOR); probs = np.array(list(ENV_PRIOR.values())); probs /= probs.sum()

    mode_cnt = Counter(); ocr_adp = 0; recoverable = captured = 0
    for _ in range(n_frames):
        car = rng.random() < 0.45
        env = str(rng.choice(envs, p=probs))
        if car:
            scene, _, _ = make_scene(random_plate_text(prng), rng)
            img = apply_env(scene, env, rng); motion = float(rng.uniform(0.05, 0.5))
        else:
            img = apply_env(_asphalt(640, 480, rng), env, rng); motion = float(rng.uniform(0, 0.008))
        d = sensor.step(img, motion=motion)
        mode_cnt[d.mode] += 1; ocr_adp += int(d.run_ocr)
        if car and RECOVERABLE[env]:
            recoverable += 1; captured += int(d.run_ocr and d.mode != "skip")

    def power_saved(pc, oc):
        adp = sum(mode_cnt[m] * pc[m] for m in mode_cnt) + oc * ocr_adp
        return 1 - adp / (n_frames * (pc["rgb_full"] + oc))

    default = power_saved(POWER_COST, OCR_COST)
    saves = [power_saved({"rgb_full": 1.0, "rgb_boost": 1.15, "ir": ir, "skip": sk}, oc)
             for ir in (0.4, 0.55, 0.7) for sk in (0.02, 0.05, 0.1) for oc in (0.5, 0.8, 1.2)]
    res = dict(n_frames=n_frames,
               ocr_calls_reduced=1 - ocr_adp / n_frames,          # ★ 실측 1차 지표
               coverage_retained=captured / max(1, recoverable), recoverable=recoverable,
               power_saved_default=default, power_saved_min=min(saves), power_saved_max=max(saves),
               mode_counts=dict(mode_cnt))

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax[0].bar(["OCR 호출\n감소(실측)", "회수가능\n캡처 유지"],
              [res["ocr_calls_reduced"]*100, res["coverage_retained"]*100],
              color=["#E0962A", "#2E8B4E"])
    ax[0].set_ylim(0, 100); ax[0].set_ylabel("%"); ax[0].set_title("추론 호출 ↓ · 캡처 유지")
    ax[1].bar(["전력 절감"], [default*100], color="#9aa1ac")
    ax[1].errorbar([0], [default*100],
                   yerr=[[default*100-res["power_saved_min"]*100], [res["power_saved_max"]*100-default*100]],
                   fmt="o", color="#1F2A3A", capsize=6)
    ax[1].set_ylim(0, 100); ax[1].set_ylabel("%")
    ax[1].set_title(f"전력 절감(가정 비용 의존)\n{res['power_saved_min']*100:.0f}~{res['power_saved_max']*100:.0f}%")
    fig.tight_layout(); fig.savefig(FIGS / "adaptive_tradeoff.png", dpi=130); plt.close(fig)
    return res


if __name__ == "__main__":
    c = eval_classifier()
    t = eval_tradeoff()
    print("=== (A) 환경분류 ===")
    print(f"  RF in-dist acc={c['acc_clf']:.3f}  macroF1={c['f1_clf']:.3f}")
    print(f"  RF ★OOD(2차 생성기) acc={c['acc_ood']:.3f}   ← 순환성 반박")
    print(f"  rule baseline acc={c['acc_rule']:.3f}  (n_test={c['n_test']})")
    print("=== (B) 추론예산-커버리지 ===")
    print(f"  ★OCR 호출 감소(실측)  {t['ocr_calls_reduced']*100:5.1f}%")
    print(f"  회수가능 캡처 유지     {t['coverage_retained']*100:5.1f}%  (recoverable={t['recoverable']})")
    print(f"  전력 절감(가정 의존)   {t['power_saved_default']*100:.1f}%  [범위 {t['power_saved_min']*100:.0f}~{t['power_saved_max']*100:.0f}%]")
    (DATA / "adaptive_metrics.json").write_text(
        json.dumps({"classify": c, "tradeoff": t}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved figs + data/adaptive_metrics.json")
