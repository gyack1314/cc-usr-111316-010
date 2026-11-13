"""把冻结资料装配为归因问题:形状核验、有理数解析、可信函数登记、变量重排。"""
from dataclasses import dataclass
from pathlib import Path

from analysis_data.domain import summarize
from analysis_data.io import integer, load

from .gaussian import GaussianVector
from .model import build_model
from .rationals import rational, rational_list, rational_matrix


@dataclass
class Problem:
    variables: list
    mean: list
    covariance: list
    transforms: list
    output_spec: dict
    sampler: dict
    psd_tolerance: float
    minimum_variance: float
    gaussian: GaussianVector
    model: object
    manifest_sha256: str
    path: str


def load_problem(path):
    """读取冻结资料并装配;基本形状由analysis_data核验(invalid-config在此被拒绝)。"""
    raw, sha = load(path)
    summarize(raw, Path(path).parent)
    if raw.get("kind") != "variance-game":
        raise ValueError("kind必须为variance-game")
    variables = list(raw["variables"])
    mean = rational_list(raw["mean"], label="mean")
    covariance = rational_matrix(raw["covariance"], label="covariance")
    transforms = raw["transforms"]
    if not isinstance(transforms, list) or any(not isinstance(t, str) for t in transforms):
        raise ValueError("transforms必须为字符串数组")
    sampler = dict(raw["sampler"])
    integer(sampler["seed"], 0)
    psd_tolerance = rational(raw["psd_tolerance"], label="psd_tolerance")
    if psd_tolerance <= 0.0:
        raise ValueError("psd_tolerance必须为正")
    minimum_variance = rational(raw["minimum_variance"], label="minimum_variance")
    if minimum_variance < 0.0:
        raise ValueError("minimum_variance不能为负")
    gaussian = GaussianVector(mean, covariance, psd_tolerance=psd_tolerance)
    model = build_model(variables, transforms, raw["output_function"])
    return Problem(
        variables,
        mean,
        covariance,
        list(transforms),
        raw["output_function"],
        sampler,
        psd_tolerance,
        minimum_variance,
        gaussian,
        model,
        sha,
        str(path),
    )


def reorder_problem(problem, permutation):
    """按排列重排变量顺序;联合分布不变,归因应按变量名一一对应。"""
    n = len(problem.variables)
    permutation = list(permutation)
    if sorted(permutation) != list(range(n)):
        raise ValueError("非法变量排列")
    variables = [problem.variables[p] for p in permutation]
    mean = [problem.mean[p] for p in permutation]
    covariance = [
        [problem.covariance[pi][pj] for pj in permutation] for pi in permutation
    ]
    transforms = [problem.transforms[p] for p in permutation]
    spec = dict(problem.output_spec)
    if spec.get("name") == "linear":
        spec["coefficients"] = [spec["coefficients"][p] for p in permutation]
    gaussian = GaussianVector(mean, covariance, psd_tolerance=problem.psd_tolerance)
    model = build_model(variables, transforms, spec)
    return Problem(
        variables,
        mean,
        covariance,
        transforms,
        spec,
        dict(problem.sampler),
        problem.psd_tolerance,
        problem.minimum_variance,
        gaussian,
        model,
        None,
        None,
    )
