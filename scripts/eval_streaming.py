"""#2 평가 — 실시간 조기 경보 (이벤트-구동 스트림) vs 배치(출차 후).

★ 정직성(B3 수정): 핵심 지표는 '주차 중 탐지율'과 '탐지 지연(=grace)'.
이전의 '평균 168분 선행'은 사실상 *위반 평균 지속시간*이라 탐지 영민함이 아님 →
'맥락(위반 지속)'으로만 표기하고 헤드라인에서 내림.
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
from kev.plotting import use_korean

use_korean()
from kev.config import DATA, FIGS, SEED
from kev.occupancy import simulate
from kev.streaming import StreamingMonitor

KIND_KO = {"unauthorized": "무단점유(미등록)", "sensor_fault": "센서고장(stuck)"}
KIND_COL = {"unauthorized": "#E0962A", "sensor_fault": "#C0392B"}


def main():
    evs = simulate(n_events=1400, seed=SEED)
    alerts = StreamingMonitor().run(evs)

    by_kind = {}
    for k in KIND_KO:
        ak = [a for a in alerts if a.kind == k]
        if ak:
            by_kind[k] = dict(
                n=len(ak),
                mean_delay=float(np.mean([a.delay for a in ak])),  # ★ 탐지 지연(=grace)
                in_occupancy_rate=float(
                    np.mean([a.lead > 0 for a in ak])
                ),  # 주차 중 탐지
                mean_violation_dur=float(np.mean([a.lead + a.delay for a in ak])),
            )  # 맥락
    res = dict(
        n_alerts=len(alerts),
        by_kind=by_kind,
        in_occupancy_rate=float(np.mean([a.lead > 0 for a in alerts]))
        if alerts
        else 0.0,
    )
    (DATA / "streaming_metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    # (좌) 타임라인 — 경보가 출차 전(주차 중)에 발생
    sample = sorted([a for a in alerts if a.lead > 0], key=lambda a: a.delay)[:8]
    for row, a in enumerate(sample):
        e = evs[a.event_idx]
        ax[0].barh(
            row,
            (e.end - e.start) / 60,
            left=e.start / 60,
            height=0.5,
            color=KIND_COL[a.kind],
            alpha=0.5,
        )
        ax[0].scatter(a.onset / 60, row, marker="v", color="black", zorder=3, s=40)
        ax[0].scatter(
            a.time / 60,
            row,
            marker="*",
            color=KIND_COL[a.kind],
            edgecolors="black",
            zorder=4,
            s=130,
        )
        ax[0].scatter(e.end / 60, row, marker="|", color="#555", zorder=3, s=120)
    ax[0].set_yticks(range(len(sample)))
    ax[0].set_yticklabels([KIND_KO[a.kind] for a in sample], fontsize=8)
    ax[0].set_xlabel("시각 (시)")
    ax[0].set_title("▼위반시작  ★경보(주차 중)  |출차 — 경보가 출차보다 앞섬")
    # (우) 탐지 지연(= grace, 위반 후 경보까지)
    ks = list(by_kind)
    delays = [by_kind[k]["mean_delay"] for k in ks]
    ax[1].bar([KIND_KO[k] for k in ks], delays, color=[KIND_COL[k] for k in ks])
    for i, k in enumerate(ks):
        ax[1].text(
            i, delays[i], f"{delays[i]:.0f}분", ha="center", va="bottom", fontsize=9
        )
    ax[1].set_ylabel("탐지 지연 (분, 위반→경보)")
    ax[1].set_title("위반 후 이만큼 만에 경보 (배치는 출차 후)")
    fig.suptitle(
        "② 실시간 조기 경보 — 주차 중 위반 탐지 (배치=출차 후 대비)", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(FIGS / "streaming.png", dpi=130)
    plt.close(fig)

    print("=== 실시간 조기 경보 (정직 지표) ===")
    print(
        f"  총 경보 {res['n_alerts']}건 · 주차 중 탐지율 {res['in_occupancy_rate'] * 100:.0f}%"
    )
    for k in by_kind:
        b = by_kind[k]
        print(
            f"  {KIND_KO[k]:16s} n={b['n']:3d}  탐지지연 {b['mean_delay']:6.1f}분  "
            f"(위반 평균지속 {b['mean_violation_dur']:.0f}분=맥락)"
        )


if __name__ == "__main__":
    main()
