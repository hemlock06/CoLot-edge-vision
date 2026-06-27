"""#1 번호판 다중프레임 추적 + 판독 투표.

단일 프레임 OCR은 악천후(비·눈)에서 오인식이 잦다. 한 차량을 여러 프레임에
걸쳐 관측하고 프레임별 판독을 **문자 단위 다수결**로 합의하면, 프레임마다
서로 다른 오류가 상쇄되어 정확도가 회복된다. (③ '악천후 다중판독' 정책의 실구현)
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import re

from .config import PLATE_RE
from .plate import correct_plate


def vote_chars(strings: list[str]) -> tuple[str, float]:
    """문자 단위 다수결. (합의문자열, 평균 동의율)."""
    cand = [s for s in strings if s]
    if not cand:
        return "", 0.0
    # 유효 포맷이 하나라도 있으면 그쪽만으로 투표(노이즈 억제)
    valid = [s for s in cand if re.match(PLATE_RE, s)]
    pool = valid if valid else cand
    L = Counter(len(s) for s in pool).most_common(1)[0][0]  # 다수 길이
    same = [s for s in pool if len(s) == L]
    out, agree = [], 0.0
    for i in range(L):
        ch, n = Counter(s[i] for s in same).most_common(1)[0]
        out.append(ch)
        agree += n / len(same)
    return "".join(out), agree / max(1, L)


@dataclass
class PlateVoter:
    """한 차량(트랙)의 프레임별 판독을 모아 합의."""

    track_id: int = 0
    reads: list[str] = field(default_factory=list)

    def add(self, raw_or_text: str, already_corrected: bool = False):
        s = raw_or_text if already_corrected else correct_plate(raw_or_text)[0]
        if s:
            self.reads.append(s)
        return self

    def consensus(self) -> dict:
        text, agree = vote_chars(self.reads)
        return dict(
            text=text,
            agree=agree,
            n=len(self.reads),
            valid=bool(re.match(PLATE_RE, text)),
        )
