"""② 불법주차 이상탐지 평가.
  - 이진 이상탐지: precision/recall/F1  (룰 only vs 룰+IForest)
  - 유형별 재현율: unauthorized / overstay / fault
  - 혼동행렬, IForest 점수 분포
산출: data/anomaly_metrics.json, figs/anomaly_recall.png, figs/anomaly_confusion.png
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean; use_korean()
from kev.config import DATA, FIGS, SEED, AnomalyCfg
from kev.occupancy import simulate
from kev.anomaly import ParkingAnomalyDetector, features

TYPES = ["unauthorized", "overstay", "fault"]
PRED_LABELS = ["normal", "unauthorized", "overstay", "fault", "anomaly"]


def binary(y_true_lab, preds):
    yt = [l != "normal" for l in y_true_lab]
    yp = [p != "normal" for p in preds]
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary",
                                                 zero_division=0)
    return dict(precision=p, recall=r, f1=f)


def per_type_recall(y_true_lab, preds):
    out = {}
    for t in TYPES:
        idx = [i for i, l in enumerate(y_true_lab) if l == t]
        if not idx:
            out[t] = None; continue
        out[t] = float(np.mean([preds[i] != "normal" for i in idx]))
    return out


def main():
    evs = simulate(n_events=1600, seed=SEED)
    n = len(evs); cut = int(n * 0.5)
    train, test = evs[:cut], evs[cut:]
    train_normal = [e for e in train if e.label == "normal"]
    y = [e.label for e in test]

    # 룰 only
    det_rule = ParkingAnomalyDetector(AnomalyCfg(), seed=SEED)  # not fitted
    pred_rule = [f.pred for f in det_rule.predict(test)]
    # 룰 + IForest
    det_ml = ParkingAnomalyDetector(AnomalyCfg(), seed=SEED).fit(train_normal)
    flags_ml = det_ml.predict(test)
    pred_ml = [f.pred for f in flags_ml]

    res = dict(
        rule_only=binary(y, pred_rule),
        rule_ml=binary(y, pred_ml),
        recall_rule=per_type_recall(y, pred_rule),
        recall_ml=per_type_recall(y, pred_ml),
        n_test=len(test),
        n_anom=int(sum(l != "normal" for l in y)),
    )
    # 초과주차 위반 크기(평균) — 잡은 것 기준
    over = [(e.end - e.paid_end) for e, p in zip(test, pred_ml)
            if e.label == "overstay" and p != "normal"]
    res["overstay_mean_violation_min"] = float(np.mean(over)) if over else None
    res["overstay_alert_latency_min"] = AnomalyCfg().overstay_grace_min

    (DATA / "anomaly_metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 그림 1: 유형별 재현율 (룰 vs 룰+ML) ---
    x = np.arange(len(TYPES)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rr = [res["recall_rule"][t] * 100 for t in TYPES]
    rm = [res["recall_ml"][t] * 100 for t in TYPES]
    ax.bar(x - w/2, rr, w, label="룰만", color="#C0C6CE")
    ax.bar(x + w/2, rm, w, label="룰 + IForest", color="#2E8B4E")
    for i, (a, b) in enumerate(zip(rr, rm)):
        ax.text(i - w/2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["무단점유", "초과주차", "센서고장"])
    ax.set_ylim(0, 108); ax.set_ylabel("재현율 (%)"); ax.legend()
    ax.set_title("유형별 이상 재현율 — ML이 센서고장을 보강")
    fig.tight_layout(); fig.savefig(FIGS / "anomaly_recall.png", dpi=130); plt.close(fig)

    # --- 그림 2: 혼동행렬 (true 유형 × pred) ---
    def simplify(p):  # anomaly→fault 계열로 귀속(통계이상=주로 고장)
        return p
    yt = ["normal" if l == "normal" else l for l in y]
    cm = confusion_matrix(yt, [simplify(p) for p in pred_ml],
                          labels=PRED_LABELS)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(PRED_LABELS))); ax.set_yticks(range(len(PRED_LABELS)))
    ax.set_xticklabels(PRED_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(PRED_LABELS, fontsize=8)
    for i in range(len(PRED_LABELS)):
        for j in range(len(PRED_LABELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true(label)")
    ax.set_title("불법주차 이상탐지 혼동행렬 (룰+IForest)")
    fig.tight_layout(); fig.savefig(FIGS / "anomaly_confusion.png", dpi=130); plt.close(fig)

    print("=== 이진 이상탐지 ===")
    print(f"  룰만      P={res['rule_only']['precision']:.3f} R={res['rule_only']['recall']:.3f} F1={res['rule_only']['f1']:.3f}")
    print(f"  룰+IForest P={res['rule_ml']['precision']:.3f} R={res['rule_ml']['recall']:.3f} F1={res['rule_ml']['f1']:.3f}")
    print("=== 유형별 재현율 (룰만 → 룰+ML) ===")
    for t in TYPES:
        print(f"  {t:12s} {res['recall_rule'][t]*100:5.1f}% → {res['recall_ml'][t]*100:5.1f}%")
    print(f"test={res['n_test']} anomalies={res['n_anom']} "
          f"overstay위반평균={res['overstay_mean_violation_min']}")


if __name__ == "__main__":
    main()
