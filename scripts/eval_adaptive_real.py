"""③ 환경분류 실데이터 검증 — 합성 분류기를 실 도로장면(BDD100K)에 적용 + 실 보정.
(HF dgural/bdd100k 실 주행이미지, weather/timeofday 라벨 → ③ 5클래스 매핑)

정직 측정 2단계 (검출기 전이학습과 동일 서사):
  (a) 합성 ③ → 실 val          : 도메인 갭(전이 안 됨)
  (b) 실-보정 ③(RF, 실 train)  : 같은 15피처로 실 환경 분류 가능함을 입증
산출: data/real_adaptive.json, figs/real_adaptive.png

라벨 매핑: rainy→rain · snowy→snow · foggy→fog · (clear/overcast)+night→low_light
          · (clear/overcast/partly)+daytime→day_normal.
한계: glare/backlit/overexposed는 공개 라벨 부재(합성 전용) · fog는 BDD 13장뿐(데이터 한계).
"""
from __future__ import annotations
import sys, json, csv, collections
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean; use_korean()
from kev.config import DATA, FIGS, SEED
from kev.adaptive import brightness_features, feature_vector
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

ROOT = DATA / "bdd_real"
SCENE = (640, 480)                       # 합성 작업 해상도에 정합
EVAL_CLASSES = ["day_normal", "low_light", "rain", "snow", "fog"]


def load_split():
    rows = list(csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8")))
    data = {"train": [], "val": []}
    for r in rows:
        p = ROOT / "images" / r["file"]
        if not p.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.resize(im, SCENE)
        fv = feature_vector(brightness_features(im))
        data[r["split"]].append((fv, r["kev_class"], r["file"]))
    return data


def recall_per_class(y_true, y_pred):
    out = {}
    for c in EVAL_CLASSES:
        idx = [i for i, t in enumerate(y_true) if t == c]
        if not idx:
            continue
        out[c] = sum(int(y_pred[i] == c) for i in idx) / len(idx)
    return out


def main():
    d = load_split()
    Xtr = np.array([x for x, _, _ in d["train"]]); ytr = [c for _, c, _ in d["train"]]
    Xva = np.array([x for x, _, _ in d["val"]]);   yva = [c for _, c, _ in d["val"]]
    print(f"train {len(ytr)} · val {len(yva)} · 분포 val:", dict(collections.Counter(yva)))

    # (a) 합성 ③ → 실 val (도메인 갭)
    clf_syn = joblib.load(DATA / "adaptive_clf.joblib")
    pred_syn = [str(p) for p in clf_syn.predict(Xva)]
    acc_syn = float(np.mean([pred_syn[i] == yva[i] for i in range(len(yva))]))
    rec_syn = recall_per_class(yva, pred_syn)

    # (b) 실-보정 ③ (RF, 실 train) → 실 val
    clf_real = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    clf_real.fit(Xtr, ytr)
    pred_real = [str(p) for p in clf_real.predict(Xva)]
    acc_real = float(np.mean([pred_real[i] == yva[i] for i in range(len(yva))]))
    rec_real = recall_per_class(yva, pred_real)

    # 5-fold CV (실 전체, 견고성) — fog(n=13) 제외해 안정적 헤드라인
    Xall = np.vstack([Xtr, Xva]); yall = np.array(ytr + yva)
    keep = yall != "fog"
    cv = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
                         Xall[keep], yall[keep],
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED))

    res = dict(
        n_train=len(ytr), n_val=len(yva), val_dist=dict(collections.Counter(yva)),
        synthetic_on_real_acc=acc_syn, synthetic_on_real_recall=rec_syn,
        real_calibrated_acc=acc_real, real_calibrated_recall=rec_real,
        real_cv_4cls_mean=float(cv.mean()), real_cv_4cls_std=float(cv.std()),
        note="fog n=13(BDD 한계); glare/backlit/overexposed는 실 라벨 부재(합성 전용)")
    (DATA / "real_adaptive.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 도식: (좌) per-class recall 전이 vs 실보정 / (우) 실 표본+③예측 몽타주
    fig = plt.figure(figsize=(13, 4.4))
    axb = fig.add_subplot(1, 2, 1)
    cls = [c for c in EVAL_CLASSES if c in rec_syn]
    x = np.arange(len(cls)); w = 0.38
    axb.bar(x - w/2, [rec_syn[c] for c in cls], w, label="합성 ③ → 실(전이)", color="#C0504D")
    axb.bar(x + w/2, [rec_real[c] for c in cls], w, label="실-보정 ③", color="#1F6FB2")
    axb.set_xticks(x); axb.set_xticklabels(cls, rotation=20, fontsize=9)
    axb.set_ylim(0, 1.05); axb.set_ylabel("재현율"); axb.legend(fontsize=8, loc="upper right")
    axb.set_title(f"실 도로장면 ③ — 전이 acc {acc_syn:.2f} → 실보정 {acc_real:.2f}\n"
                  f"(야간 전이 {rec_syn.get('low_light',0):.2f} ↔ 주간/날씨 약함)", fontsize=10)
    # 표본 4장: 실라벨 vs 합성 ③ 예측 (야간 적중·주간→rain 오인 가시화)
    pick, seen = [], collections.Counter()
    for fv, c, fn in d["val"]:
        if seen[c] < 1 and c in ("day_normal", "low_light", "rain", "snow"):
            pick.append((fn, c)); seen[c] += 1
        if len(pick) >= 4: break
    slots = [3, 4, 7, 8]                        # 2×4 그리드 우측 2열
    for (fn, c), slot in zip(pick, slots):
        im = cv2.resize(cv2.imread(str(ROOT / "images" / fn)), SCENE)
        pr = str(clf_syn.predict(feature_vector(brightness_features(im))[None, :])[0])
        a = fig.add_subplot(2, 4, slot); a.axis("off")
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ok = "✓" if pr == c else "✗"
        a.set_title(f"실:{c} · ③:{pr} {ok}", fontsize=8,
                    color="#1F6FB2" if pr == c else "#C0504D")
    fig.suptitle("③ 환경분류 실데이터 검증 (BDD100K 실 주행) — 저조도는 전이, 날씨는 합성 전용",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "real_adaptive.png", dpi=130); plt.close(fig)

    print("=== ③ 실데이터 검증 (BDD100K 실 도로장면) ===")
    print(f"  (a) 합성 ③ → 실 val  : acc {acc_syn:.3f}  recall {rec_syn}")
    print(f"  (b) 실-보정 ③ → 실 val: acc {acc_real:.3f}  recall {rec_real}")
    print(f"  실 5-fold CV(4클래스, fog 제외): {cv.mean():.3f} ± {cv.std():.3f}")


if __name__ == "__main__":
    main()
