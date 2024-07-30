'''
@File: ssm_methods.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn

import torch.nn.functional as F

from einops import rearrange,repeat

import selective_scan_cuda
'''
how to get selective_scan_cuda
https://github.com/state-spaces/mamba/issues/97
casual_cnn:https://github.com/Dao-AILab/causal-conv1d/releases
mamba-ssm: https://github.com/state-spaces/mamba/releases
'''

def selective_scan_ref(u,delta,A,B,C,D,z,delta_bias,delta_softplus=True,return_last_state=False):
    '''
    选择性扫描,for循环效率低仅供参考
    :param u:b,d_in,l
    :param delta:b,d_in,l
    :param A:d_in,d_s
    :param B:b,d_s,l
    :param C:b,d_s,l
    :param D:d_in
    :param z:b,d_in,l
    :param delta_bias:d_in
    :param delta_softplus:
    :param return_last_state:bool
    :return:b,d,l
    '''
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta+delta_bias[...,None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate,length = u.shape[0], A.shape[0], A.shape[1],u.shape[2]
    # is_variable_B = B.dim() >= 3
    # is_variable_C = C.dim() >= 3
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    # x = A.new_zeros((batch, dim, length, dstate))
    # y = torch.zeros_like(x)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    last_state = None
    # x = torch.einsum('bdln,bdln->bdln',deltaA,x)+deltaB_u
    # y = torch.einsum('bdln,bnl->bdl', x,C)
    last_state = x[:,:,-1]
    #循环导致计算效率变慢
    for i in range(u.shape[-1]):
        x = deltaA[:,:,i]*x+deltaB_u[:,:,i]
        y = torch.einsum('bdn,bn->bd',x,C[:,:,i])
        if i == u.shape[2]-1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys,dim=2)# (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out,last_state)


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)
def selective_scan_fn(u,delta,A,B,C,D,z,delta_bias,delta_softplus=True,return_last_state=False):
    # return selective_scan_ref(u,delta,A,B,C,D,z,delta_bias,delta_softplus,return_last_state)
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


