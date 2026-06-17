"""matplotlib 한글 폰트 설정."""
import matplotlib
from matplotlib import font_manager


def use_korean():
    for name, path in [("Malgun Gothic", r"C:\Windows\Fonts\malgun.ttf"),
                       ("NanumGothic", None), ("Gulim", r"C:\Windows\Fonts\gulim.ttc")]:
        try:
            if path:
                font_manager.fontManager.addfont(path)
            matplotlib.rcParams["font.family"] = name
            break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
