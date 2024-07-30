'''
@File: caucnn.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn.functional as  F

def causal_conv1d_ref(x,weight,bias=None,activation=None):
    '''
    wrap vanilla CNN
    :param x:b,d,l
    :param weight:d,l
    :param bias:d
    :param activation:None
    :return:b,d,l
    '''
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    out = out[..., :seqlen]
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)

def causal_conv1d_fn(x,weight,bias=None,activation=None):
    return causal_conv1d_ref(x, weight, bias, activation)