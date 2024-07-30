'''
@File: rmsnorm.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 21, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
def rms_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False):
    dtype = x.dtype
    residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)


def rms_norm_fn(x, weight, bias, residual=None, prenorm=False,eps=1e-6):
    return rms_norm_ref(x, weight, bias, residual, eps, prenorm)

class RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.register_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False):
        return rms_norm_fn(x,self.weight,self.bias,residual=residual,eps=self.eps,prenorm=prenorm)


def layer_norm_ref(x,weight,bias,residual=None,eps=1e-6,prenorm=False):
    dtype = x.dtype
    residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype),x.shape[-1:],weight=weight,bias=bias,eps=eps)
    return out if not prenorm else (out,x)

def layer_norm_fn(x, weight, bias, residual=None, prenorm=False,eps=1e-6):
    return layer_norm_ref(x, weight, bias, residual, eps, prenorm)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x