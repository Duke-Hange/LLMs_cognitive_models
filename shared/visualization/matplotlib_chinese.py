"""
统一设置 matplotlib 中文字体，避免中文和特殊符号（如负号）显示为方框。
在任意绘图脚本中 import 并调用 setup_chinese_font() 即可。
"""

import os
import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _get_chinese_font_path():
    """根据系统返回中文字体文件路径（若存在）。"""
    system = platform.system()
    if system == "Windows":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        candidates = [
            os.path.join(windir, "Fonts", "msyh.ttc"),   # Microsoft YaHei
            os.path.join(windir, "Fonts", "msyhbd.ttc"), # Microsoft YaHei Bold
            os.path.join(windir, "Fonts", "simhei.ttf"), # SimHei
            os.path.join(windir, "Fonts", "simsun.ttc"), # SimSun
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
    elif system == "Darwin":
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
    else:
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
    return None


def setup_chinese_font():
    """
    设置 matplotlib 使用支持中文的字体，并修复负号显示为方框的问题。
    建议在每次绘图前调用（或脚本开头 import 后调用一次）。
    """
    # 负号使用 Unicode 减号，避免显示为方框
    plt.rcParams["axes.unicode_minus"] = False

    font_path = _get_chinese_font_path()
    if font_path:
        try:
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            family = prop.get_name()
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans", "Arial"]
            return
        except Exception:
            pass

    # 无本地字体文件时，使用字体族名（依赖系统已安装字体）
    if platform.system() == "Windows":
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "SimSun", "FangSong",
            "DejaVu Sans", "Arial",
        ]
    elif platform.system() == "Darwin":
        plt.rcParams["font.sans-serif"] = [
            "PingFang SC", "Heiti SC", "Arial Unicode MS",
            "DejaVu Sans", "Arial",
        ]
    else:
        plt.rcParams["font.sans-serif"] = [
            "WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans", "Arial",
        ]
