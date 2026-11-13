"""潜在高斯向量的联合与条件抽样器,与正式输出函数解耦。

奇异协方差按像空间与广义逆处理:先校验对称性、半正定容差与条件点支持;
零特征值方向不加噪声,给定完全线性相关变量后剩余不确定性正确消失。
随机流显式传入(random.Random),另留种子便捷版本;协方差分解保留在
decomposition 属性中供复用与留证。
"""
import math
import random

from .linalg import decompose_psd, matmul, norm, submatrix, transpose


class GaussianVector:
    """N(mean, covariance),协方差可半正定且奇异。"""

    def __init__(self, mean, covariance, *, psd_tolerance, label="潜在高斯"):
        self.mean = [float(m) for m in mean]
        n = len(self.mean)
        if len(covariance) != n or any(len(row) != n for row in covariance):
            raise ValueError(f"{label}协方差形状与均值不一致")
        self.covariance = [list(map(float, row)) for row in covariance]
        self.psd_tolerance = float(psd_tolerance)
        self.decomposition = decompose_psd(self.covariance, self.psd_tolerance, label=label)

    @property
    def dimension(self):
        return len(self.mean)

    @property
    def rank(self):
        return self.decomposition.rank

    def sample_joint(self, n, rng):
        """联合抽样 n 点;零特征值方向严格无噪声,样本落在分布支撑仿射包上。"""
        dec = self.decomposition
        d = self.dimension
        out = []
        for _ in range(n):
            x = list(self.mean)
            for f in dec.factors:
                xi = rng.gauss(0.0, 1.0)
                for j in range(d):
                    x[j] += xi * f[j]
            out.append(x)
        return out

    def sample_joint_seeded(self, n, seed):
        """随机流的种子便捷版本。"""
        return self.sample_joint(n, random.Random(seed))

    def marginal(self, subset):
        idx = tuple(sorted(subset))
        return GaussianVector(
            [self.mean[i] for i in idx],
            submatrix(self.covariance, idx, idx),
            psd_tolerance=self.psd_tolerance,
            label="边际高斯",
        )

    def conditional(self, subset):
        """预计算 X_T | X_S 的条件抽样器(线性代数确定,可跨样本设计复用)。"""
        return ConditionalGaussian(self, tuple(sorted(subset)))


class ConditionalGaussian:
    """X_T | X_S=x_S:均值 μ_T+Σ_TS·Σ_SS⁺·(x_S−μ_S),协方差 Schur 补;条件点须在 Σ_SS 像空间。"""

    def __init__(self, gaussian, subset):
        d = gaussian.dimension
        self.subset = tuple(sorted(subset))
        self.target = tuple(i for i in range(d) if i not in self.subset)
        self.gaussian = gaussian
        cov = gaussian.covariance
        tol = gaussian.psd_tolerance
        self.mean_target = [gaussian.mean[i] for i in self.target]
        self.mean_subset = [gaussian.mean[i] for i in self.subset]
        self.max_support_residual = 0.0
        self.support_tolerance = math.sqrt(tol)
        if not self.subset:
            self.decomposition_ss = None
            self.gain = []
            self.conditional_decomposition = gaussian.decomposition
            return
        self.decomposition_ss = decompose_psd(
            submatrix(cov, self.subset, self.subset), tol, label="Σ_SS"
        )
        if not self.target:
            self.gain = []
            self.conditional_decomposition = None
            return
        sigma_ts = submatrix(cov, self.target, self.subset)
        sigma_st = transpose(sigma_ts)
        self.gain = matmul(sigma_ts, self.decomposition_ss.pinv_matrix())
        sigma_tt = submatrix(cov, self.target, self.target)
        schur = matmul(self.gain, sigma_st)
        cond_cov = [
            [sigma_tt[i][j] - schur[i][j] for j in range(len(self.target))]
            for i in range(len(self.target))
        ]
        self.conditional_decomposition = decompose_psd(cond_cov, tol, label="条件协方差")

    def check_support(self, x_s):
        """校验条件点支持:x_S−μ_S 须在 Σ_SS 像空间内,否则拒绝。"""
        if not self.subset:
            return
        b = [float(x_s[k]) - self.mean_subset[k] for k in range(len(self.subset))]
        residual = self.decomposition_ss.support_residual(b)
        if residual > self.max_support_residual:
            self.max_support_residual = residual
        lam_max = self.decomposition_ss.eigenvalues[0] if self.decomposition_ss.eigenvalues else 0.0
        limit = self.support_tolerance * (math.sqrt(lam_max) + norm(b))
        if residual > limit:
            raise ValueError(
                f"条件点不在Σ_SS像空间: 残差{residual:.3e}超过{limit:.3e},该点不在分布支撑上"
            )

    def conditional_mean(self, x_s):
        self.check_support(x_s)
        if not self.target:
            return []
        if not self.subset:
            return list(self.mean_target)
        b = [float(x_s[k]) - self.mean_subset[k] for k in range(len(self.subset))]
        shift = [sum(row[k] * b[k] for k in range(len(b))) for row in self.gain]
        return [self.mean_target[j] + shift[j] for j in range(len(self.target))]

    def sample(self, x_s, n, rng):
        """给定 x_S 抽 n 个补集样本;条件协方差秩为零时剩余不确定性消失。"""
        mean = self.conditional_mean(x_s)
        return self.draw_around(mean, n, rng)

    def sample_seeded(self, x_s, n, seed):
        return self.sample(x_s, n, random.Random(seed))

    def draw_around(self, mean, n, rng):
        dec = self.conditional_decomposition
        if dec is None:
            return [[] for _ in range(n)]
        out = []
        for _ in range(n):
            x = list(mean)
            for f in dec.factors:
                xi = rng.gauss(0.0, 1.0)
                for j in range(len(mean)):
                    x[j] += xi * f[j]
            out.append(x)
        return out

    def evidence(self):
        """留证:子集、秩、截断负特征值、最大支持残差。"""
        out = {
            "subset": list(self.subset),
            "target": list(self.target),
            "max_support_residual": self.max_support_residual,
        }
        if self.decomposition_ss is not None:
            out["rank_ss"] = self.decomposition_ss.rank
            out["clipped_negatives_ss"] = list(self.decomposition_ss.clipped_negatives)
        if self.conditional_decomposition is not None:
            out["rank_conditional"] = self.conditional_decomposition.rank
            out["clipped_negatives_conditional"] = list(
                self.conditional_decomposition.clipped_negatives
            )
        return out
