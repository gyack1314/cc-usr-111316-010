# 联合不确定性资料

variables顺序对应潜在高斯X。transforms只允许identity或exp，可信输出函数linear(coefficients,constant)、product(全部变换量相乘)和weighted-sum-products(明确项)均由本地登记，不执行输入代码。条件子集指潜在X坐标；exp可逆，因此可等价解释物理量。

协方差对称且半正定，完全重复变量作为两个不同分摊者保留。对S条件均值mu_T+Sigma_TS*pinv(Sigma_SS)*(xS-muS)，协方差Sigma_TT-Sigma_TS*pinv(Sigma_SS)*Sigma_ST，条件点须在Sigma_SS像空间。数值截断阈值、因子秩及条件协方差负残差均应留证，显著不定拒绝。

每外层条件点各取两批条件独立样本得到m1,m2，用交叉协方差估Var(E[Y|XS])，不是m1样本方差；分母Var(Y)与其估计设计单列。多次独立随机化估整体误差，同时通过固定外层增加内层、固定子集增加排列的对照拆出主要误差来源，不把三项强行假定独立后相加。S空/全集用定义值，有效方差为0则整体undefined。读取工具只检查矩阵和预算形状。
