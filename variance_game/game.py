"""条件方差博弈 v(S)=Var(E[Y|X_S])/Var(Y) 的两层蒙特卡洛估计。

同一外层条件点取两批条件独立内层样本(m1,m2),以交叉协方差估 Var(E[Y|X_S]),
对内层噪声无偏;不能把含噪条件均值 m1 直接取样本方差(含内层噪声上偏)。
外层云在同一样本设计的各子集间共享;v 缓存以(设计标识,子集)为键,
不同样本设计不混用。空集 v=0、全集 v=1 用定义值,不消耗模型求值。
"""
import math
import random
from dataclasses import dataclass, field


class ZeroVarianceError(ValueError):
    """Var(Y)有效值为零,条件方差博弈未定义。"""


@dataclass
class VRecord:
    subset: tuple
    value: float
    definitional: bool
    outer: int = 0
    inner: int = 0
    se_total: float = 0.0
    se_inner: float = 0.0
    se_outer_residual: float = 0.0
    max_support_residual: float = 0.0


@dataclass
class DenominatorRecord:
    mean: float
    variance: float
    samples: int
    seed: int


class DesignContext:
    """一次样本设计:自己的随机流、外层云、分母矩与 v 缓存;缓存键含设计标识。"""

    def __init__(self, game, design_id, seeds, outer, inner):
        self.game = game
        self.design_id = design_id
        self.seeds = dict(seeds)
        self.outer = outer
        self.inner = inner
        self.cache = {}
        self._outer_cloud = None
        self._denominator = None
        self._inner_master = random.Random(seeds["inner_master"])

    def next_inner_seed(self):
        return self._inner_master.getrandbits(63)

    def outer_cloud(self):
        if self._outer_cloud is None:
            self._outer_cloud = self.game.gaussian.sample_joint(
                self.outer, random.Random(self.seeds["outer"])
            )
        return self._outer_cloud


class ConditionalVarianceGame:
    """v(S) 估计器;线性代数(条件算子)跨设计复用,样本缓存按设计隔离。"""

    def __init__(self, gaussian, model, *, outer, inner, minimum_variance):
        if outer < 2:
            raise ValueError("外层样本数至少为2")
        if inner < 2:
            raise ValueError("内层样本数至少为2(需估计内层噪声)")
        self.gaussian = gaussian
        self.model = model
        self.outer = outer
        self.inner = inner
        self.minimum_variance = float(minimum_variance)
        self._conditionals = {}

    @property
    def dimension(self):
        return self.gaussian.dimension

    def conditional(self, subset):
        key = tuple(sorted(subset))
        if key not in self._conditionals:
            self._conditionals[key] = self.gaussian.conditional(key)
        return self._conditionals[key]

    def conditionals_evidence(self):
        return [cond.evidence() for cond in self._conditionals.values()]

    def new_design(self, design_id, seed, *, inner=None):
        root = random.Random(seed)
        seeds = {
            k: root.getrandbits(63)
            for k in ("outer", "inner_master", "denominator", "permutation", "control")
        }
        return DesignContext(
            self, design_id, seeds, self.outer, inner if inner is not None else self.inner
        )

    def inner_control_design(self, ctx, factor=4):
        """固定外层云与分母、内层加量并更换内层随机流的对照设计。"""
        seeds = dict(ctx.seeds)
        seeds["inner_master"] = ctx.seeds["control"]
        return DesignContext(
            self, f"{ctx.design_id}|innerx{factor}", seeds, ctx.outer, ctx.inner * factor
        )

    def denominator(self, ctx):
        """分母 Var(Y) 的独立估计设计,与 v(S) 分子设计单列。"""
        if ctx._denominator is None:
            rng = random.Random(ctx.seeds["denominator"])
            m = self.outer * self.inner
            ys = [self.model(x) for x in self.gaussian.sample_joint(m, rng)]
            mean = sum(ys) / m
            var = sum((y - mean) ** 2 for y in ys) / (m - 1)
            ctx._denominator = DenominatorRecord(mean, var, m, ctx.seeds["denominator"])
        return ctx._denominator

    def estimate_v(self, subset, ctx):
        key = tuple(sorted(subset))
        if key in ctx.cache:
            return ctx.cache[key]
        if not key:
            rec = VRecord(key, 0.0, True)
        elif len(key) == self.dimension:
            rec = VRecord(key, 1.0, True)
        else:
            rec = self._estimate_v_mc(key, ctx)
        ctx.cache[key] = rec
        return rec

    def _estimate_v_mc(self, key, ctx):
        cond = self.conditional(key)
        den = self.denominator(ctx)
        if den.variance <= self.minimum_variance:
            raise ZeroVarianceError(
                f"Var(Y)有效值为零({den.variance:.3e}),条件方差博弈未定义"
            )
        cloud = ctx.outer_cloud()
        ra = random.Random(ctx.next_inner_seed())
        rb = random.Random(ctx.next_inner_seed())
        k = ctx.outer
        n = ctx.inner
        idx_t = list(cond.target)
        m1, m2, t1, t2 = [], [], [], []
        for x in cloud:
            x_s = [x[i] for i in key]
            mean = cond.conditional_mean(x_s)  # 含像空间校验
            for m_out, t_out, rng in ((m1, t1, ra), (m2, t2, rb)):
                total = 0.0
                total2 = 0.0
                for comp in cond.draw_around(mean, n, rng):
                    full = list(x)
                    for pos, j in enumerate(idx_t):
                        full[j] = comp[pos]
                    y = self.model(full)
                    total += y
                    total2 += y * y
                m_out.append(total / n)
                var_batch = max((total2 - total * total / n) / (n - 1), 0.0)
                t_out.append(var_batch / n)
        kf = float(k)
        s1, s2 = sum(m1), sum(m2)
        s12 = sum(a * b for a, b in zip(m1, m2))
        u = (s12 - s1 * s2 / kf) / (kf - 1.0)  # 交叉协方差,对内层噪声无偏
        bar1, bar2 = s1 / kf, s2 / kf
        ds = [(a - bar1) * (b - bar2) for a, b in zip(m1, m2)]
        sd = sum(ds)
        var_d = max((sum(x * x for x in ds) - sd * sd / kf) / (kf - 1.0), 0.0)
        se_total = math.sqrt(kf * var_d) / (kf - 1.0)
        mid = (bar1 + bar2) / 2.0
        shrink = (kf - 1.0) / kf
        iota = 0.0
        for a, b, ta, tb in zip(m1, m2, t1, t2):
            m_tilde = (a + b) / 2.0 - mid
            tau2 = (ta + tb) / 2.0
            iota += 2.0 * m_tilde * m_tilde * tau2 * shrink + ta * tb * shrink * shrink
        se_inner = math.sqrt(iota) / (kf - 1.0)
        se_outer = math.sqrt(max(se_total * se_total - se_inner * se_inner, 0.0))
        var_y = den.variance
        return VRecord(
            key,
            u / var_y,
            False,
            k,
            n,
            se_total / var_y,
            se_inner / var_y,
            se_outer / var_y,
            cond.max_support_residual,
        )
