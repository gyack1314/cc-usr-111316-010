"""仅核对输入形状，不生成分析结果。"""
from .io import integer
def summarize(v,root):
    n=len(v['variables'])
    if len(set(v['variables']))!=n or len(v['mean'])!=n or len(v['covariance'])!=n or any(len(r)!=n for r in v['covariance']):raise ValueError('联合矩阵形状错误')
    for k in ('replicates','outer','inner','permutations'):integer(v['sampler'][k],1)
    return {'variables':n,'replicates':v['sampler']['replicates'],'outer_samples':v['sampler']['outer']}
