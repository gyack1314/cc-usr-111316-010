# 相关排放因子的条件方差归因与误差拆分

本项目保存气候行动研究的合成资料与领域口径。`analysis_data.io`读取JSON及核验原始文件身份，`analysis_data.domain`核对基本形状，`analysis_data.cli`展示数量。详细量纲、时间、契约和统计口径见[资料说明](docs/data-contract.md)。地区、家庭、合同、价格及排放数据均为合成，不代表真实政策评价或投资建议。

Python 3.11以上，无第三方依赖。项目根目录可运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
```

`inspect`只展示资料，不做拟合、求解或分析发布。`invalid-config.json`包含基本配置错误，其余清单在身份上完整，可能包含需领域分析判断的不可识别、无界、冲突或不适用情况。有理数以字符串保留，null不等于0，随机种子属于分析设计。

Dockerfile固定Linux镜像；`docker build -t climate-analysis .`构建，`docker run --rm climate-analysis`显示摘要，`docker run --rm climate-analysis python -m unittest discover -s tests -v`运行已有读取检查。资料均可本地读取，运行不需要线上账户。

## 归因引擎 `variance_game`

在`analysis_data`读取核验之上，`variance_game`按资料契约执行条件方差博弈的Shapley归因（统计分摊，不是干预因果效果），纯标准库实现：

- `gaussian`：潜在高斯联合/条件抽样器，与输出函数解耦。协方差可半正定且奇异；条件均值用广义逆`mu_T+Sigma_TS*pinv(Sigma_SS)*(xS-muS)`，条件协方差为Schur补；条件点先做Sigma_SS像空间校验，零特征值方向不加噪声，给定完全线性相关变量后剩余不确定性消失。对称性、截断阈值、因子秩、条件协方差负残差均留证，显著不定拒绝。
- `model`：登记变换（identity/exp）与可信输出函数（linear、product、weighted-sum-products），规格即数据，不执行输入代码。
- `game`：`v(S)=Var(E[Y|X_S])/Var(Y)`两层蒙特卡洛。同一外层条件点取两批条件独立内层样本m1、m2，以交叉协方差估计分子（对内层噪声无偏），不是对含噪条件均值取样本方差；空集0、全集1用定义值；分母Var(Y）与其估计设计单列；缓存以（设计标识，子集）为键，不同样本设计不混用；固定种子可重现。
- `shapley`：变量少枚举子集，变量多抽排列并复用前缀缓存；排列预算达到d!时枚举全排列，与枚举模式精确一致。完全重复的变量作为不同分摊者对称分摊，不按列顺序归给先选中者。
- `attribution`：整体误差来自多次独立随机化（重复）；内层误差由"固定外层与分母、内层×4"对照，排列误差由"固定子集值、排列×4"对照，三项不假定独立强行相加。蒙特卡洛小负边际原样保留，不截断为0也不重新归一；报告sum-to-one残差；预算用完且误差仍大于`--se-tolerance`则`status=not-converged`；Var(Y）有效值为0则`status=undefined`。
- `analytic`：线性高斯输出与全exp乘积输出的闭式v(S）与Shapley值，用于核对蒙特卡洛引擎。

运行示例：

```sh
python -m variance_game.cli run data/normal.json --analytic
python -m variance_game.cli run data/lognormal-emission-product.json --analytic --reorder-check
python -m unittest discover -s tests -v
```

`data/lognormal-emission-product.json`为相关对数正态排放乘积演示：燃料量与排放因子相关，燃料量另有一条完全重复记录（协方差奇异，秩4/5），两个重复记录作为不同分摊者对称分摊。`--analytic`给出闭式Shapley对照；`--reorder-check`把变量逆序重排后重跑，核对归因按变量名对应且差异在合并标准误约3倍内。
