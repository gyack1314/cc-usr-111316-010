"""命令行:python -m variance_game.cli run 资料.json [--se-tolerance T] [--analytic] [--reorder-check]"""
import argparse
import json

from .analytic import analytic_shapley, analytic_v_linear, analytic_v_product
from .attribution import run_attribution
from .problem import load_problem, reorder_problem
from .rationals import rational_list


def analytic_section(problem):
    """当输出属于登记的闭式情形时给出解析Shapley对照。"""
    spec = problem.output_spec
    d = len(problem.variables)
    g = problem.gaussian
    if spec.get("name") == "linear" and all(t == "identity" for t in problem.transforms):
        coef = rational_list(spec["coefficients"], label="linear.coefficients")
        try:
            phi = analytic_shapley(d, lambda s: analytic_v_linear(g, coef, s))
        except ValueError as exc:
            return {"kind": "linear-gaussian", "note": str(exc)}
        return {"kind": "linear-gaussian", "phi": phi}
    if spec.get("name") == "product" and all(t == "exp" for t in problem.transforms):
        phi = analytic_shapley(d, lambda s: analytic_v_product(g, s))
        return {"kind": "lognormal-product", "phi": phi}
    return {"kind": None, "note": "该输出函数与变换组合无登记闭式解析"}


def reorder_check(problem, report, se_tolerance):
    """变量重排后再跑一遍,核对归因按变量名对应且误差含义一致。"""
    d = len(problem.variables)
    permutation = list(reversed(range(d)))
    variant = reorder_problem(problem, permutation)
    followup = run_attribution(
        variant.gaussian,
        variant.model,
        variant.sampler,
        minimum_variance=variant.minimum_variance,
        se_tolerance=se_tolerance,
    )
    if report["attribution"] is None or followup["attribution"] is None:
        return {"note": "归因未定义,跳过重排核对", "status": followup["status"]}
    rows = []
    max_ratio = 0.0
    for i in range(d):
        j = permutation.index(i)
        est_a = report["attribution"][i]["estimate"]
        est_b = followup["attribution"][j]["estimate"]
        se_a = report["attribution"][i]["se_total"] or 0.0
        se_b = followup["attribution"][j]["se_total"] or 0.0
        combined = (se_a ** 2 + se_b ** 2) ** 0.5
        delta = abs(est_a - est_b)
        ratio = delta / combined if combined > 0.0 else 0.0
        max_ratio = max(max_ratio, ratio)
        rows.append(
            {
                "variable": problem.variables[i],
                "estimate_original": est_a,
                "estimate_reordered": est_b,
                "abs_delta": delta,
                "combined_se": combined,
                "delta_in_se": ratio,
            }
        )
    return {
        "permutation": permutation,
        "rows": rows,
        "max_delta_in_se": max_ratio,
        "consistent": max_ratio <= 3.0,
        "note": "重排后同一变量的归因差应在合并标准误约3倍内,且标准误量级一致",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("file")
    parser.add_argument("--se-tolerance", type=float, default=0.05)
    parser.add_argument("--analytic", action="store_true")
    parser.add_argument("--reorder-check", action="store_true")
    args = parser.parse_args()
    problem = load_problem(args.file)
    report = run_attribution(
        problem.gaussian,
        problem.model,
        problem.sampler,
        minimum_variance=problem.minimum_variance,
        se_tolerance=args.se_tolerance,
    )
    out = {
        "file": args.file,
        "manifest_sha256": problem.manifest_sha256,
        "report": report,
    }
    if args.analytic:
        section = analytic_section(problem)
        if section.get("phi") and report.get("attribution"):
            diffs = [
                abs(a["estimate"] - b)
                for a, b in zip(report["attribution"], section["phi"])
            ]
            section["max_abs_diff_vs_estimate"] = max(diffs)
        out["analytic"] = section
    if args.reorder_check:
        out["reorder_check"] = reorder_check(problem, report, args.se_tolerance)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
