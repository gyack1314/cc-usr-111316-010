"""条件方差博弈Shapley归因引擎的核对测试。"""
import json
import math
import random
import unittest

from variance_game.analytic import analytic_shapley, analytic_v_linear, analytic_v_product
from variance_game.attribution import run_attribution
from variance_game.game import ConditionalVarianceGame, ZeroVarianceError
from variance_game.gaussian import GaussianVector
from variance_game.linalg import decompose_psd, jacobi_eigen
from variance_game.model import build_model
from variance_game.problem import load_problem, reorder_problem
from variance_game.rationals import rational
from variance_game.shapley import choose_mode, shapley_exact, shapley_permutations

TOL = 1e-10


def small_sampler(**kw):
    s = {"seed": 2026, "replicates": 6, "outer": 48, "inner": 16, "permutations": 32}
    s.update(kw)
    return s


class RationalTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(rational("1/2"), 0.5)
        self.assertEqual(rational("-3/4"), -0.75)
        self.assertEqual(rational("2"), 2.0)
        self.assertEqual(rational("0"), 0.0)

    def test_reject(self):
        for bad in (None, True, 0.5, "abc", "1/0"):
            with self.assertRaises(ValueError):
                rational(bad)


class LinalgTests(unittest.TestCase):
    def test_jacobi_known(self):
        values, vectors, ev = jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])
        self.assertAlmostEqual(values[0], 3.0, places=12)
        self.assertAlmostEqual(values[1], 1.0, places=12)
        self.assertAlmostEqual(abs(vectors[0][0]), math.sqrt(0.5), places=12)
        self.assertLessEqual(ev.off_residual, 1e-12)

    def test_decompose_singular(self):
        dec = decompose_psd([[1.0, 1.0], [1.0, 1.0]], TOL, label="t")
        self.assertEqual(dec.rank, 1)
        self.assertEqual(dec.clipped_negatives, [])

    def test_decompose_rejects(self):
        with self.assertRaises(ValueError):
            decompose_psd([[1.0, 2.0], [2.0, 1.0]], TOL, label="不定")
        with self.assertRaises(ValueError):
            decompose_psd([[1.0, 0.5], [0.4, 1.0]], TOL, label="不对称")

    def test_pinv_and_support(self):
        dec = decompose_psd([[1.0, 1.0], [1.0, 1.0]], TOL, label="t")
        out = dec.pinv_matvec([2.0, 2.0])
        self.assertAlmostEqual(out[0], 1.0, places=12)
        self.assertAlmostEqual(out[1], 1.0, places=12)
        self.assertAlmostEqual(dec.support_residual([1.0, 1.0]), 0.0, places=12)
        self.assertAlmostEqual(dec.support_residual([1.0, -1.0]), math.sqrt(2.0), places=12)


class GaussianTests(unittest.TestCase):
    def setUp(self):
        self.g = GaussianVector([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], psd_tolerance=TOL)
        self.dup = GaussianVector([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]], psd_tolerance=TOL)

    def test_joint_moments(self):
        xs = self.g.sample_joint_seeded(4000, 7)
        m0 = sum(x[0] for x in xs) / len(xs)
        c01 = sum(x[0] * x[1] for x in xs) / len(xs)
        self.assertAlmostEqual(m0, 0.0, delta=0.06)
        self.assertAlmostEqual(c01, 0.5, delta=0.06)

    def test_zero_eigenvalue_no_noise(self):
        for x in self.dup.sample_joint_seeded(50, 3):
            self.assertEqual(x[0], x[1])  # 零特征值方向严格无噪声

    def test_conditional_degenerates(self):
        cond = self.dup.conditional((0,))
        self.assertEqual(cond.conditional_decomposition.rank, 0)  # 剩余不确定性消失
        samples = cond.sample_seeded([0.7], 5, 11)
        for s in samples:
            self.assertAlmostEqual(s[0], 0.7, places=12)

    def test_support_check(self):
        g3 = GaussianVector(
            [0.0] * 3,
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            psd_tolerance=TOL,
        )
        cond = g3.conditional((0, 1))
        cond.check_support([0.3, 0.3])  # 在像空间内
        with self.assertRaises(ValueError):
            cond.check_support([0.3, -0.7])  # 不在支撑上,拒绝

    def test_conditional_mean_matches_formula(self):
        cond = self.g.conditional((0,))
        mu = cond.conditional_mean([2.0])
        self.assertAlmostEqual(mu[0], 0.5 * 2.0, places=12)


class ModelTests(unittest.TestCase):
    def test_linear(self):
        m = build_model(["a", "b"], ["identity", "identity"],
                        {"name": "linear", "coefficients": ["2", "1/2"], "constant": "1"})
        self.assertEqual(m([3.0, 4.0]), 9.0)
        self.assertEqual(m.evaluations, 1)

    def test_product_and_exp(self):
        m = build_model(["a", "b"], ["exp", "exp"], {"name": "product"})
        self.assertAlmostEqual(m([0.0, 0.0]), 1.0)
        self.assertAlmostEqual(m([1.0, 2.0]), math.exp(3.0))

    def test_weighted_sum_products(self):
        m = build_model(["a", "b"], ["identity", "identity"], {
            "name": "weighted-sum-products",
            "terms": [
                {"coefficient": "3/2", "variables": ["a", "b"]},
                {"coefficient": "1", "variables": ["b"]},
            ],
            "constant": "1/2",
        })
        self.assertEqual(m([2.0, 3.0]), 0.5 + 1.5 * 6.0 + 3.0)

    def test_unregistered_rejected(self):
        with self.assertRaises(ValueError):
            build_model(["a"], ["identity"], {"name": "eval-this"})
        with self.assertRaises(ValueError):
            build_model(["a"], ["sqrt"], {"name": "product"})


class GameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = load_problem("data/normal.json")

    def game(self, **kw):
        s = small_sampler(**kw)
        return ConditionalVarianceGame(
            self.problem.gaussian, self.problem.model,
            outer=s["outer"], inner=s["inner"],
            minimum_variance=self.problem.minimum_variance,
        ), s

    def test_definitional_endpoints(self):
        game, s = self.game()
        ctx = game.new_design("t", s["seed"])
        before = self.problem.model.evaluations
        self.assertEqual(game.estimate_v((), ctx).value, 0.0)
        self.assertEqual(game.estimate_v((0, 1), ctx).value, 1.0)
        self.assertTrue(game.estimate_v((), ctx).definitional)
        self.assertEqual(self.problem.model.evaluations, before)  # 定义值不耗预算

    def test_cross_covariance_matches_analytic(self):
        game, s = self.game(outer=256, inner=16)
        ctx = game.new_design("t", s["seed"])
        rec = game.estimate_v((0,), ctx)
        self.assertAlmostEqual(rec.value, 0.75, delta=max(4 * rec.se_total, 0.06))
        self.assertGreater(rec.se_inner, 0.0)

    def test_cache_distinguishes_designs(self):
        game, s = self.game()
        c1 = game.new_design("d1", 1)
        c2 = game.new_design("d2", 2)
        r1 = game.estimate_v((0,), c1)
        r2 = game.estimate_v((0,), c2)
        self.assertIs(c1.cache[(0,)], r1)
        self.assertIs(c2.cache[(0,)], r2)
        self.assertNotEqual(r1.value, r2.value)  # 不同样本设计不混用缓存

    def test_zero_variance_raises(self):
        problem = load_problem("data/constant-output-undefined.json")
        game = ConditionalVarianceGame(
            problem.gaussian, problem.model, outer=8, inner=4,
            minimum_variance=problem.minimum_variance,
        )
        with self.assertRaises(ZeroVarianceError):
            game.estimate_v((0,), game.new_design("t", 5))


class ShapleyTests(unittest.TestCase):
    def test_choose_mode(self):
        self.assertEqual(choose_mode(2, 64), "exact")
        self.assertEqual(choose_mode(5, 64), "exact")
        self.assertEqual(choose_mode(10, 64), "permutation")

    def test_negative_marginals_not_clipped(self):
        values = {(): 0.0, (0,): -0.2, (1,): 0.5, (0, 1): 1.0}
        res = shapley_exact(2, lambda s: values[s])
        self.assertAlmostEqual(res.phi[0], 0.15)
        self.assertAlmostEqual(res.phi[1], 0.85)
        self.assertIn(-0.2, res.marginals)  # 负边际原样保留
        self.assertAlmostEqual(sum(res.phi), 1.0)  # 不重新归一,效率性天然成立

    def test_permutation_full_enumeration_equals_exact(self):
        def v(s):
            return (sum((i + 1) * 0.3 for i in s)) ** 2 / 10 + 0.01 * len(s)
        exact = shapley_exact(4, v)
        perm = shapley_permutations(4, v, 24, random.Random(0))
        for a, b in zip(exact.phi, perm.phi):
            self.assertAlmostEqual(a, b, places=12)

    def test_prefix_reuse_cache(self):
        calls = []
        cache = {}

        def v(s):
            if s not in cache:
                calls.append(s)
                cache[s] = len(s) * 0.1 + sum(s) * 0.01
            return cache[s]

        shapley_permutations(4, v, 10, random.Random(1))
        self.assertLessEqual(len(calls), 2 ** 4)  # 前缀复用:不超过全部子集
        before = len(calls)
        shapley_permutations(4, v, 10, random.Random(1))  # 同一批排列
        self.assertEqual(len(calls), before)  # 命中缓存不再求值

    def test_permutation_se_nonzero(self):
        def v(s):
            return sum((i + 1) ** 2 for i in s) / 100.0 + (len(s) % 2) * 0.3
        res = shapley_permutations(5, v, 8, random.Random(3))
        self.assertTrue(any(s and s > 0 for s in res.se_permutation))


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normal = load_problem("data/normal.json")
        cls.singular = load_problem("data/singular-duplicate-variables.json")
        cls.constant = load_problem("data/constant-output-undefined.json")
        cls.lognormal = load_problem("data/lognormal-emission-product.json")

    def attribute(self, problem, sampler):
        return run_attribution(
            problem.gaussian, problem.model, sampler,
            minimum_variance=problem.minimum_variance,
        )

    def test_linear_gaussian_matches_analytic(self):
        s = dict(self.normal.sampler)
        report = self.attribute(self.normal, s)
        self.assertEqual(report["status"], "converged")
        self.assertEqual(report["mode"], "exact")
        analytic = analytic_shapley(
            2, lambda sub: analytic_v_linear(self.normal.gaussian, [1.0, 1.0], sub)
        )
        self.assertEqual(analytic, [0.5, 0.5])
        for est, truth, row in zip(analytic, [0.5, 0.5], report["attribution"]):
            self.assertAlmostEqual(row["estimate"], truth, delta=max(4 * row["se_total"], 0.02))
        self.assertLess(report["sum_to_one_residual"], 1e-12)
        self.assertEqual(report["variance_of_Y"]["design"]["samples"], 128 * 32)

    def test_duplicate_variables_split_symmetrically(self):
        report = self.attribute(self.singular, dict(self.singular.sampler))
        rows = report["attribution"]
        self.assertAlmostEqual(rows[0]["estimate"], 0.5, places=12)
        self.assertAlmostEqual(rows[1]["estimate"], 0.5, places=12)
        self.assertAlmostEqual(rows[0]["estimate"], rows[1]["estimate"], places=12)
        # 不是按列顺序把全部贡献归给先选中的变量
        self.assertLess(abs(rows[0]["estimate"] - 1.0), 0.75)

    def test_constant_output_undefined(self):
        report = self.attribute(self.constant, dict(self.constant.sampler))
        self.assertEqual(report["status"], "undefined")
        self.assertIsNone(report["attribution"])

    def test_invalid_config_rejected(self):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            load_problem("data/invalid-config.json")

    def test_reproducible_fixed_seed(self):
        s = small_sampler()
        r1 = self.attribute(self.normal, s)
        r2 = self.attribute(self.normal, s)
        self.assertEqual(json.dumps(r1, sort_keys=True), json.dumps(r2, sort_keys=True))

    def test_not_converged_flag(self):
        report = run_attribution(
            self.normal.gaussian, self.normal.model, small_sampler(),
            minimum_variance=self.normal.minimum_variance, se_tolerance=1e-9,
        )
        self.assertEqual(report["status"], "not-converged")
        self.assertTrue(report["convergence"]["budget_exhausted"])

    def test_lognormal_product_matches_analytic(self):
        s = small_sampler(seed=99, replicates=8, outer=64, inner=16)
        report = self.attribute(self.lognormal, s)
        analytic = analytic_shapley(
            5, lambda sub: analytic_v_product(self.lognormal.gaussian, sub)
        )
        self.assertAlmostEqual(analytic[0], analytic[1], places=15)  # 重复变量解析对称
        for row, truth in zip(report["attribution"], analytic):
            self.assertAlmostEqual(
                row["estimate"], truth, delta=max(4 * row["se_total"], 0.03)
            )
        # 重复变量在蒙特卡洛下也近似对称
        rows = report["attribution"]
        self.assertAlmostEqual(rows[0]["estimate"], rows[1]["estimate"], delta=0.05)

    def test_reorder_consistency(self):
        s = small_sampler(seed=77, replicates=6, outer=48, inner=16)
        base = self.attribute(self.lognormal, s)
        permutation = list(reversed(range(5)))
        variant = reorder_problem(self.lognormal, permutation)
        follow = self.attribute(variant, s)
        # 解析层重排不变性(精确)
        a1 = analytic_shapley(5, lambda sub: analytic_v_product(self.lognormal.gaussian, sub))
        a2 = analytic_shapley(5, lambda sub: analytic_v_product(variant.gaussian, sub))
        for i in range(5):
            j = permutation.index(i)
            self.assertAlmostEqual(a1[i], a2[j], places=12)
            delta = abs(base["attribution"][i]["estimate"] - follow["attribution"][j]["estimate"])
            combined = (base["attribution"][i]["se_total"] ** 2
                        + follow["attribution"][j]["se_total"] ** 2) ** 0.5
            self.assertLessEqual(delta, 4 * combined + 1e-9)

    def test_evidence_and_budget_recorded(self):
        report = self.attribute(self.singular, small_sampler())
        self.assertEqual(report["evidence"]["gaussian"]["rank"], 1)
        self.assertTrue(report["evidence"]["conditionals"])
        self.assertGreater(report["budget"]["model_evaluations"], 0)
        self.assertEqual(len(report["evidence"]["design_seeds"]), small_sampler()["replicates"])

    def test_permutation_mode_integration(self):
        g = GaussianVector(
            [0.0] * 6,
            [[1.0 if i == j else 0.2 for j in range(6)] for i in range(6)],
            psd_tolerance=TOL,
        )
        model = build_model(
            [f"x{i}" for i in range(6)], ["identity"] * 6,
            {"name": "linear", "coefficients": ["1", "2", "3", "1", "2", "3"], "constant": "0"},
        )
        s = small_sampler(seed=5, replicates=4, outer=24, inner=8, permutations=8)
        report = run_attribution(g, model, s, minimum_variance=1e-12)
        self.assertEqual(report["mode"], "permutation")
        self.assertTrue(any(x and x > 0 for x in report["controls"]["permutation"]["se_at_budget"]))
        self.assertLess(report["sum_to_one_residual"], 1e-12)
        self.assertAlmostEqual(sum(r["estimate"] for r in report["attribution"]), 1.0)


if __name__ == "__main__":
    unittest.main()
