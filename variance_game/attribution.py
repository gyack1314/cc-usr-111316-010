"""归因装配:重复随机化估整体误差,对照实验拆误差来源,收敛判定与报告。

整体误差来自多次独立随机化(重复);内层误差由"固定外层与分母、内层加量"
对照,排列误差由"固定子集值、排列加量"对照;三项不假定独立强行相加。
预算用完且误差仍大则明确标记未稳定;Var(Y)有效值为零则整体未定义。
"""
import math
import random

from .game import ConditionalVarianceGame
from .shapley import choose_mode, shapley_exact, shapley_permutations

_ERROR_SEMANTICS = (
    "se_total来自重复随机化(估计值为各重复均值);se_inner为主设计内层噪声传播,"
    "se_outer_residual为总方差减内层方差的残差诊断,se_permutation为排列抽样误差;"
    "三者为主要误差来源的对照拆分,不假定独立强行相加"
)


def _engine(d, mode, permutations, value, seed_perm):
    if mode == "exact":
        return shapley_exact(d, value)
    return shapley_permutations(d, value, permutations, random.Random(seed_perm))


def run_attribution(gaussian, model, sampler, *, minimum_variance, se_tolerance=0.05):
    """运行完整归因,返回JSON可序列化报告。sampler含seed/replicates/outer/inner/permutations。"""
    d = gaussian.dimension
    evaluations_before = model.evaluations
    game = ConditionalVarianceGame(
        gaussian,
        model,
        outer=sampler["outer"],
        inner=sampler["inner"],
        minimum_variance=minimum_variance,
    )
    mode = choose_mode(d, sampler["permutations"])
    root = random.Random(sampler["seed"])
    design_seeds = [root.getrandbits(63) for _ in range(sampler["replicates"])]
    primary = game.new_design("replicate-0", design_seeds[0])
    den = game.denominator(primary)
    report = {
        "mode": mode,
        "variables": list(getattr(model, "variables", [f"x{i}" for i in range(d)])),
        "sampler": dict(sampler),
        "se_tolerance": se_tolerance,
        "variance_of_Y": {
            "estimate": den.variance,
            "mean": den.mean,
            "design": {"seed": den.seed, "samples": den.samples},
            "minimum_variance": minimum_variance,
        },
        "error_semantics": _ERROR_SEMANTICS,
    }
    if den.variance <= minimum_variance:
        report.update(
            {
                "status": "undefined",
                "reason": f"Var(Y)有效值为零({den.variance:.3e}≤{minimum_variance:.1e}),归因未定义",
                "attribution": None,
                "sum_to_one_residual": None,
            }
        )
        return report

    results = []
    contexts = [primary]
    for r in range(1, len(design_seeds)):
        contexts.append(game.new_design(f"replicate-{r}", design_seeds[r]))
    for ctx in contexts:
        value = lambda subset, c=ctx: game.estimate_v(subset, c).value
        results.append(_engine(d, mode, sampler["permutations"], value, ctx.seeds["permutation"]))
    result_primary = results[0]
    n_rep = len(results)

    estimates = [sum(res.phi[i] for res in results) / n_rep for i in range(d)]
    if n_rep > 1:
        rep_std = [
            math.sqrt(sum((res.phi[i] - estimates[i]) ** 2 for res in results) / (n_rep - 1))
            for i in range(d)
        ]
        se_total = [s / math.sqrt(n_rep) for s in rep_std]
    else:
        rep_std = [None] * d
        se_total = [None] * d

    se_outer = [0.0] * d
    se_inner = [0.0] * d
    for i in range(d):
        for subset, coeff in result_primary.coefficients[i].items():
            rec = primary.cache[subset]
            se_outer[i] += (coeff * rec.se_outer_residual) ** 2
            se_inner[i] += (coeff * rec.se_inner) ** 2
        se_outer[i] = math.sqrt(se_outer[i])
        se_inner[i] = math.sqrt(se_inner[i])
    se_perm = result_primary.se_permutation

    residual = max(abs(sum(res.phi) - 1.0) for res in results)
    negatives = [m for m in result_primary.marginals if m < 0.0]

    # 对照实验一:固定外层云与分母,内层×4并更换内层随机流
    control_ctx = game.inner_control_design(primary)
    control_value = lambda subset: game.estimate_v(subset, control_ctx).value
    control_result = _engine(
        d, mode, sampler["permutations"], control_value, control_ctx.seeds["permutation"]
    )
    inner_control = {
        "design": control_ctx.design_id,
        "inner": control_ctx.inner,
        "note": "固定外层云与分母,内层样本×4、内层随机流更换;差异量级即内层噪声",
        "max_abs_delta": max(
            abs(a - b) for a, b in zip(control_result.phi, result_primary.phi)
        ),
        "delta": [a - b for a, b in zip(control_result.phi, result_primary.phi)],
    }
    # 对照实验二:固定子集值(主设计缓存),排列加量
    if mode == "permutation":
        value_primary = lambda subset: game.estimate_v(subset, primary).value
        res_m = shapley_permutations(
            d, value_primary, sampler["permutations"], random.Random(primary.seeds["control"])
        )
        res_4m = shapley_permutations(
            d,
            value_primary,
            4 * sampler["permutations"],
            random.Random(primary.seeds["control"] + 1),
        )
        permutation_control = {
            "note": "固定子集值,排列×4;排列标准误应约减半",
            "se_at_budget": res_m.se_permutation,
            "se_at_4x": res_4m.se_permutation,
            "max_abs_delta": max(abs(a - b) for a, b in zip(res_m.phi, res_4m.phi)),
        }
    else:
        permutation_control = {
            "mode": "exact",
            "se_permutation": [0.0] * d,
            "note": "子集枚举精确,排列抽样误差为0",
        }

    max_se = max((s for s in se_total if s is not None), default=None)
    converged = max_se is not None and max_se <= se_tolerance
    attribution = [
        {
            "variable": report["variables"][i],
            "estimate": estimates[i],
            "se_total": se_total[i],
            "replicate_std": rep_std[i],
            "se_outer_residual": se_outer[i],
            "se_inner": se_inner[i],
            "se_permutation": se_perm[i],
        }
        for i in range(d)
    ]
    report.update(
        {
            "status": "converged" if converged else "not-converged",
            "attribution": attribution,
            "sum_to_one_residual": residual,
            "negative_marginals": {
                "count": len(negatives),
                "min": min(negatives) if negatives else 0.0,
                "note": "蒙特卡洛小负边际原样保留,未截断为0,未重新归一",
            },
            "controls": {"inner": inner_control, "permutation": permutation_control},
            "convergence": {
                "se_tolerance": se_tolerance,
                "max_se_total": max_se,
                "budget_exhausted": True,
                "note": "预算已用完;误差仍大于容差则未稳定" if not converged else "误差在容差内",
            },
            "budget": {
                "model_evaluations": model.evaluations - evaluations_before,
                "replicates": sampler["replicates"],
                "outer": sampler["outer"],
                "inner": sampler["inner"],
                "permutations": sampler["permutations"],
            },
            "evidence": {
                "gaussian": {
                    "dimension": d,
                    "rank": gaussian.rank,
                    "psd_tolerance": gaussian.psd_tolerance,
                    "symmetry_residual": gaussian.decomposition.symmetry_residual,
                    "clipped_negatives": list(gaussian.decomposition.clipped_negatives),
                    "jacobi": {
                        "sweeps": gaussian.decomposition.eigen.sweeps,
                        "off_residual": gaussian.decomposition.eigen.off_residual,
                    },
                },
                "conditionals": game.conditionals_evidence(),
                "design_seeds": design_seeds,
            },
        }
    )
    return report
