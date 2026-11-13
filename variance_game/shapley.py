"""条件方差博弈的Shapley归因:少变量枚举子集,多变量抽排列并复用前缀缓存。

Shapley值是统计分摊(效率性:Σφ=v(全集)−v(空集)=1),不是干预因果效果;
完全重复的变量作为不同分摊者对称分摊,不按列顺序把相关贡献归给先选中的一方。
蒙特卡洛造成的小负边际原样保留,不截断为0也不重新归一。
"""
import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations, permutations


@dataclass
class ShapleyResult:
    mode: str
    phi: list
    coefficients: list  # 每个分摊者: {子集: 系数},用于误差传播
    marginals: list  # 全部原始边际贡献(含负值,不截断)
    per_permutation: list  # 每个排列的φ向量(枚举模式为None)
    se_permutation: list
    permutations: int


def choose_mode(d, permutations):
    """变量少时枚举子集,变量多时抽排列:按 2^d 与 排列预算×d 的成本比较选择。"""
    return "exact" if 2 ** d <= max(1, permutations) * max(1, d) else "permutation"


def shapley_exact(d, value):
    """枚举全部子集;value(排序元组)→v。权重用Fraction精确表示。"""
    if d < 1:
        raise ValueError("至少一个变量")
    phi = [0.0] * d
    coeffs = [dict() for _ in range(d)]
    marginals = []
    for i in range(d):
        others = [j for j in range(d) if j != i]
        for s in range(d):
            w = float(Fraction(math.factorial(s) * math.factorial(d - 1 - s), math.factorial(d)))
            for combo in combinations(others, s):
                base = tuple(sorted(combo))
                key = tuple(sorted(combo + (i,)))
                marg = value(key) - value(base)
                marginals.append(marg)
                phi[i] += w * marg
                coeffs[i][key] = coeffs[i].get(key, 0.0) + w
                coeffs[i][base] = coeffs[i].get(base, 0.0) - w
    return ShapleyResult("exact", phi, coeffs, marginals, None, [0.0] * d, math.factorial(d))


def shapley_permutations(d, value, m, rng):
    """抽 m 个排列,沿前缀走并复用缓存;排列数≥d!时枚举全排列,与枚举模式精确一致。"""
    if m < 1:
        raise ValueError("排列数至少为1")
    total = math.factorial(d)
    if m >= total:
        perms = list(permutations(range(d)))
    else:
        perms = [tuple(rng.sample(range(d), d)) for _ in range(m)]
    count = len(perms)
    accum = [0.0] * d
    per_perm = []
    coeffs = [dict() for _ in range(d)]
    marginals = []
    for pi in perms:
        pre = ()
        pphi = [0.0] * d
        for i in pi:
            key = tuple(sorted(pre + (i,)))
            marg = value(key) - value(pre)
            marginals.append(marg)
            pphi[i] = marg
            accum[i] += marg
            coeffs[i][key] = coeffs[i].get(key, 0.0) + 1.0 / count
            coeffs[i][pre] = coeffs[i].get(pre, 0.0) - 1.0 / count
            pre = key
        per_perm.append(pphi)
    phi = [a / count for a in accum]
    se = []
    for i in range(d):
        if count > 1:
            mu = phi[i]
            var = sum((p[i] - mu) ** 2 for p in per_perm) / (count - 1)
            se.append(math.sqrt(var / count))
        else:
            se.append(None)
    return ShapleyResult("permutation", phi, coeffs, marginals, per_perm, se, count)
