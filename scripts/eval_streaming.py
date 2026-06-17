"""#2 평가 — 실시간 조기 경보 vs 배치(출차 후) 판정.
산출: data/streaming_metrics.json, figs/streaming.png
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kev.plotting import use_korean; use_korean()
from kev.config import DATA, FIGS, SEED
from kev.occupancy import simulate
from kev.streaming import StreamingMonitor

KIND_KO = {"unauthorized": "무단점유", "overstay": "초과주차", "sensor_fault": "센서고장(stuck)"}
KIND_COL = {"unauthorized": "#E0962A", "overstay": "#7B5EA7", "sensor_fault": "#C0392B"}


def main():
    evs = simulate(n_events=1400, seed=SEED)
    mon = StreamingMonitor()
    alerts = mon.run(evs)

    by_kind = {}
    for k in KIND_KO:
        ak = [a for a in alerts if a.kind == k]
        if ak:
            by_kind[k] = dict(n=len(ak),
                              mean_delay=float(np.mean([a.delay for a in ak])),
                              mean_lead=float(np.mean([a.lead for a in ak])),
                              early_rate=float(np.mean([a.lead > 0 for a in ak])))
    res = dict(n_alerts=len(alerts), by_kind=by_kind,
               mean_lead_all=float(np.mean([a.lead for a in alerts])) if alerts else 0.0)
    (DATA / "streaming_metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 타임라인: 위반 8건 (점유 막대 + 위반시작 ▼ + 경보 ★ + 출차) ---
    sample = sorted([a for a in alerts if a.lead > 0],
                    key=lambda a: a.delay)[:8]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for row, a in enumerate(sample):
        e = evs[a.event_idx]
        ax[0].barh(row, (e.end - e.start) / 60, left=e.start / 60, height=0.5,
                   color=KIND_COL[a.kind], alpha=0.5)
        ax[0].scatter(a.onset / 60, row, marker="v", color="black", zorder=3, s=40)
        ax[0].scatter(a.time / 60, row, marker="*", color=KIND_COL[a.kind],
                      edgecolors="black", zorder=4, s=130)
        ax[0].scatter(e.end / 60, row, marker="|", color="#555", zorder=3, s=120)
    ax[0].set_yticks(range(len(sample)))
    ax[0].set_yticklabels([KIND_KO[a.kind] for a in sample], fontsize=8)
    ax[0].set_xlabel("시각 (시)")
    ax[0].set_title("▼위반시작  ★경보(주차중!)  |출차 — 경보가 출차보다 앞섬")

    # --- 막대: 종류별 평균 선행시간(출차까지 여유) ---
    ks = list(by_kind); lead = [by_kind[k]["mean_lead"] for k in ks]
    ax[1].bar([KIND_KO[k] for k in ks], lead, color=[KIND_COL[k] for k in ks])
    for i, k in enumerate(ks):
        ax[1].text(i, lead[i], f"{lead[i]:.0f}분", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylabel("평균 선행시간 (분)")
    ax[1].set_title("배치(출차 후) 대비 — 이만큼 일찍 경보")

    fig.suptitle("② 실시간 조기 경보 — 차량 주차 중 위반 즉시 탐지", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "streaming.png", dpi=130); plt.close(fig)

    print("=== 실시간 조기 경보 ===")
    print(f"  총 경보 {res['n_alerts']}건, 평균 선행시간 {res['mean_lead_all']:.0f}분")
    for k in by_kind:
        b = by_kind[k]
        print(f"  {KIND_KO[k]:16s} n={b['n']:3d}  지연 {b['mean_delay']:5.1f}분  "
              f"선행 {b['mean_lead']:6.1f}분  주차중탐지 {b['early_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
