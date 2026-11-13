"""对称矩阵的纯Python线性代数:Jacobi特征分解、半正定分解留证、广义逆与像空间残差。"""
import math
from dataclasses import dataclass, field


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def norm(x):
    return math.sqrt(dot(x, x))


def matvec(a, x):
    return [dot(row, x) for row in a]


def matmul(a, b):
    bt = list(zip(*b))
    return [[dot(row, col) for col in bt] for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def submatrix(a, rows, cols):
    return [[a[i][j] for j in cols] for i in rows]


def quad(a, x):
    """xᵀAx。"""
    return dot(x, matvec(a, x))


@dataclass
class EigenEvidence:
    sweeps: int
    off_residual: float
    tolerance: float


@dataclass
class PsdDecomposition:
    """对称半正定分解 Σ=QΛQᵀ 与留证:仅正特征值方向保留因子,零特征值方向不产生噪声。"""

    eigenvalues: list
    eigenvectors: list
    rank: int
    tolerance: float
    symmetry_residual: float
    clipped_negatives: list
    eigen: EigenEvidence
    factors: list = field(default_factory=list)

    def pinv_matvec(self, b):
        """广义逆作用 Σ⁺b,只在像空间内求解。"""
        out = [0.0] * len(b)
        for i in range(self.rank):
            q = self.eigenvectors[i]
            c = dot(q, b) / self.eigenvalues[i]
            for j in range(len(out)):
                out[j] += c * q[j]
        return out

    def pinv_matrix(self):
        n = len(self.eigenvalues)
        out = [[0.0] * n for _ in range(n)]
        for i in range(self.rank):
            q = self.eigenvectors[i]
            w = 1.0 / self.eigenvalues[i]
            for r in range(n):
                qr = w * q[r]
                for c in range(n):
                    out[r][c] += qr * q[c]
        return out

    def support_residual(self, b):
        """b 到像空间(列空间)的残差范数;零特征值方向上的分量必须为零。"""
        r = list(b)
        for i in range(self.rank):
            q = self.eigenvectors[i]
            c = dot(q, b)
            for j in range(len(r)):
                r[j] -= c * q[j]
        return norm(r)


def jacobi_eigen(matrix, *, tolerance=1e-12, max_sweeps=64):
    """循环Jacobi对称特征分解;返回(降序特征值, 对应特征向量列, 留证)。"""
    n = len(matrix)
    a = [list(map(float, row)) for row in matrix]
    v = [[float(i == j) for j in range(n)] for i in range(n)]
    scale = math.sqrt(sum(x * x for row in a for x in row)) or 1.0

    def off_norm():
        return math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(i + 1, n)))

    sweeps = 0
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        if off_norm() <= tolerance * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= tolerance * scale * 1e-3:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0.0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: -a[i][i])
    values = [a[i][i] for i in order]
    vectors = [[v[r][i] for r in range(n)] for i in order]
    return values, vectors, EigenEvidence(sweeps=sweeps, off_residual=off_norm(), tolerance=tolerance)


def decompose_psd(matrix, tolerance, *, label="协方差"):
    """对称半正定分解并留证:对称性校验、显著不定拒绝、负特征值截断记录、因子秩。"""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError(f"{label}必须为同维方阵")
    scale = max(1.0, max(abs(x) for row in matrix for x in row))
    asym = max(abs(matrix[i][j] - matrix[j][i]) for i in range(n) for j in range(n))
    if asym > tolerance * scale:
        raise ValueError(f"{label}不对称: 残差{asym:.3e}超过容差{tolerance:.1e}")
    sym = [[0.5 * (matrix[i][j] + matrix[j][i]) for j in range(n)] for i in range(n)]
    values, vectors, evidence = jacobi_eigen(sym, tolerance=min(tolerance, 1e-12))
    lam_scale = max(1.0, max(abs(x) for x in values))
    if values[-1] < -tolerance * lam_scale:
        raise ValueError(f"{label}显著不定: 最小特征值{values[-1]:.3e}低于-{tolerance:.1e}·谱范数")
    clipped = [x for x in values if x < 0.0]
    values = [x if x > 0.0 else 0.0 for x in values]
    rank = sum(1 for x in values if x > tolerance * lam_scale)
    factors = [[math.sqrt(values[i]) * vectors[i][j] for j in range(n)] for i in range(rank)]
    return PsdDecomposition(
        eigenvalues=values,
        eigenvectors=vectors,
        rank=rank,
        tolerance=tolerance,
        symmetry_residual=asym,
        clipped_negatives=clipped,
        eigen=evidence,
        factors=factors,
    )
