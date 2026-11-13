"""相关高斯输入的条件方差博弈Shapley归因(统计分摊,非干预因果效果)。"""
from .analytic import analytic_shapley, analytic_v_linear, analytic_v_product
from .attribution import run_attribution
from .game import ConditionalVarianceGame, DesignContext, VRecord, ZeroVarianceError
from .gaussian import ConditionalGaussian, GaussianVector
from .model import build_model
from .problem import load_problem, reorder_problem
from .shapley import choose_mode, shapley_exact, shapley_permutations

__all__ = [
    "GaussianVector",
    "ConditionalGaussian",
    "build_model",
    "ConditionalVarianceGame",
    "DesignContext",
    "VRecord",
    "ZeroVarianceError",
    "choose_mode",
    "shapley_exact",
    "shapley_permutations",
    "run_attribution",
    "load_problem",
    "reorder_problem",
    "analytic_v_linear",
    "analytic_v_product",
    "analytic_shapley",
]
