"""有理数字符串解析:资料以字符串保留有理数,null不等于0。"""
import math
from fractions import Fraction


def rational(value, *, label="有理数"):
    """字符串有理数(如"1/2"、"-3")解析为float;拒绝null、布尔、非字符串与非有限值。"""
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"{label}必须为字符串有理数,收到: {value!r}")
    try:
        out = float(Fraction(value.strip()))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{label}无法解析: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label}非有限值: {value!r}")
    return out


def rational_list(values, *, label):
    if not isinstance(values, list):
        raise ValueError(f"{label}必须为数组")
    return [rational(v, label=f"{label}[{i}]") for i, v in enumerate(values)]


def rational_matrix(rows, *, label):
    if not isinstance(rows, list):
        raise ValueError(f"{label}必须为矩阵")
    return [rational_list(r, label=f"{label}[{i}]") for i, r in enumerate(rows)]
