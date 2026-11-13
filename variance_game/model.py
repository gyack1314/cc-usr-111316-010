"""可信输出函数与变换登记:规格即数据,只认登记名,不执行输入代码。

变换只含 identity/exp;输出函数只含 linear、product、weighted-sum-products。
物理正量由 exp 变换产生;exp 可逆,对潜在 X 坐标取条件等价于对物理量取条件。
"""
import math

from .rationals import rational, rational_list

_TRANSFORMS = {"identity": lambda z: z, "exp": math.exp}


class Model:
    """变换+可信输出函数;evaluations 记录求值次数用于预算留证。"""

    def __init__(self, variables, transforms, output_spec):
        n = len(variables)
        if not isinstance(transforms, list) or len(transforms) != n:
            raise ValueError("变换数量与变量不一致")
        unknown = [t for t in transforms if t not in _TRANSFORMS]
        if unknown:
            raise ValueError(f"未登记变换: {unknown}")
        if not isinstance(output_spec, dict):
            raise ValueError("输出函数规格必须为对象")
        self.variables = list(variables)
        self.transforms = list(transforms)
        self.output_spec = output_spec
        self.evaluations = 0
        self._tf = [_TRANSFORMS[t] for t in transforms]
        name = output_spec.get("name")
        if name == "linear":
            extra = set(output_spec) - {"name", "coefficients", "constant"}
            if extra:
                raise ValueError(f"linear含未登记字段: {sorted(extra)}")
            coef = rational_list(output_spec["coefficients"], label="linear.coefficients")
            if len(coef) != n:
                raise ValueError("linear系数长度与变量不一致")
            const = rational(output_spec.get("constant", "0"), label="linear.constant")
            self._fn = lambda z: const + sum(c * v for c, v in zip(coef, z))
        elif name == "product":
            extra = set(output_spec) - {"name"}
            if extra:
                raise ValueError(f"product含未登记字段: {sorted(extra)}")
            self._fn = lambda z: math.prod(z)
        elif name == "weighted-sum-products":
            extra = set(output_spec) - {"name", "terms", "constant"}
            if extra:
                raise ValueError(f"weighted-sum-products含未登记字段: {sorted(extra)}")
            terms = output_spec["terms"]
            if not isinstance(terms, list) or not terms:
                raise ValueError("weighted-sum-products.terms必须为非空数组")
            index = {v: i for i, v in enumerate(variables)}
            parsed = []
            for k, term in enumerate(terms):
                coeff = rational(term["coefficient"], label=f"terms[{k}].coefficient")
                names = term["variables"]
                if not isinstance(names, list) or not names:
                    raise ValueError(f"terms[{k}].variables必须为非空数组")
                if any(v not in index for v in names):
                    raise ValueError(f"terms[{k}]引用未登记变量: {names}")
                parsed.append((coeff, tuple(index[v] for v in names)))
            const = rational(
                output_spec.get("constant", "0"), label="weighted-sum-products.constant"
            )
            self._fn = lambda z: const + sum(
                c * math.prod(z[i] for i in idxs) for c, idxs in parsed
            )
        else:
            raise ValueError(f"未登记输出函数: {name!r}")

    def __call__(self, x):
        self.evaluations += 1
        return self._fn([tf(v) for tf, v in zip(self._tf, x)])


def build_model(variables, transforms, output_spec):
    return Model(variables, transforms, output_spec)
