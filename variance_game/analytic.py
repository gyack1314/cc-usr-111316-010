"""解析核对:线性高斯输出与全exp乘积输出的 v(S) 闭式解,用于核对蒙特卡洛引擎。

线性(全identity变换, Y=cᵀX+b): v(S)=cᵀΣ_{·S}Σ_SS⁺Σ_{S·}c / cᵀΣc。
乘积(全exp变换, Y=exp(1ᵀX)): 记 a=1向量, ω_S=bᵀΣ_SS b, b=a_S+Gᵀa_T,
G=Σ_TSΣ_SS⁺ 为条件增益,则 v(S)=(e^{ω_S}−1)/(e^{ω_full}−1)。
"""
import math

from .linalg import dot, quad, submatrix
from .shapley import shapley_exact


def analytic_v_linear(gaussian, coefficients, subset):
    key = tuple(sorted(subset))
    d = gaussian.dimension
    if not key:
        return 0.0
    if len(key) == d:
        return 1.0
    cov = gaussian.covariance
    den = quad(cov, coefficients)
    if den <= 0.0:
        raise ValueError("cᵀΣc=0,线性输出方差为零,解析v未定义")
    g = [sum(cov[s][j] * coefficients[j] for j in range(d)) for s in key]
    cond = gaussian.conditional(key)
    return dot(g, cond.decomposition_ss.pinv_matvec(g)) / den


def analytic_v_product(gaussian, subset):
    key = tuple(sorted(subset))
    d = gaussian.dimension
    if not key:
        return 0.0
    if len(key) == d:
        return 1.0
    cov = gaussian.covariance
    omega_full = quad(cov, [1.0] * d)
    cond = gaussian.conditional(key)
    b = [
        1.0 + sum(cond.gain[t][j] for t in range(len(cond.target)))
        for j in range(len(key))
    ]
    omega = quad(submatrix(cov, key, key), b)
    return math.expm1(omega) / math.expm1(omega_full)


def analytic_shapley(d, v_of_subset):
    """由解析 v(S) 枚举出解析 Shapley 值。"""
    return shapley_exact(d, v_of_subset).phi
