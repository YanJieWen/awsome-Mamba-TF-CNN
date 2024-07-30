'''
@File: bi_mamba.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 20, 2024
@HomePage: https://github.com/YanJieWen
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import repeat,rearrange


from ..ssm_zoo.bi_selective_scan_interface import bimamba_inner_fn,mamba_inner_fn_no_out_proj,mamba_inner_fn
from ..blocks.caucnn import causal_conv1d_fn
from ..ssm_zoo.ssm_methods import selective_scan_fn

class Mamba(nn.Module):
    def __init__(self,
                 d_model:int=768,
                 d_state:int=16,
                 d_conv:int=4,
                 expand:int=2,
                 dt_rank:str='auto',
                 dt_min:float=0.001,
                 dt_max:float=0.01,
                 dt_init:str='random',
                 dt_scale:float=1.0,
                 dt_init_floor:float=1e-4,
                 conv_bias:bool=True,
                 bias:bool=False,
                 use_fast_path:bool=True,
                 layer_idx:int=None,
                 bimamba_type:str='none',
                 if_divide_out:bool=False
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand*self.d_model)
        self.dt_rank = math.ceil(self.d_model/16) if dt_rank=='auto' else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_divide_out =if_divide_out

        self.in_proj = nn.Linear(self.d_model,self.d_inner*2,bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner,self.d_inner,bias=conv_bias,kernel_size=d_conv,
                                groups=self.d_inner,padding=d_conv-1)#套入因果卷积，普通卷积长度会发生变化
        self.x_proj = nn.Linear(self.d_inner,self.dt_rank+self.d_state*2,bias=False)
        self.dt_proj = nn.Linear(self.dt_rank,self.d_inner,bias=True)
        self.activation = "silu"
        self.act = nn.SiLU()
        #initial
        dt_init_std = self.dt_rank**-0.5*dt_scale
        if dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight,dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner,) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(torch.arange(1,self.d_state+1,dtype=torch.float32),'n->d n',d=self.d_inner).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        if bimamba_type=='v1':
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type=='v2': #V1版本仅有矩阵A在前后方向不是共享的，V2版本所有的参数均非独立的
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv-1
            )
            self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner,self.d_model,bias=bias)

    def forward(self,hidden_states):
        '''

        :param hidden_states: b,m+1,d
        :return:same as hidden_states
        '''
        batch,seq_len,dim = hidden_states.shape
        conv_state,ssm_state = None,None
        # xz = rearrange(self.in_proj.weight@rearrange(hidden_states,'b l d->d (b l)'),'d (b l)->b d l',l=seq_len)
        # if self.in_proj.bias is not None:
        #     xz = xz+rearrange(self.in_proj.bias.to(dtype=xz.dtype),'d->d 1')
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seq_len,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        if self.use_fast_path:
            if self.bimamba_type == 'v1':
                A_b = -torch.exp(self.A_b_log.float())
                out = bimamba_inner_fn(
                    xz,self.conv1d.weight,self.conv1d.bias,
                    self.x_proj.weight,self.dt_proj.weight,
                    self.out_proj.weight,self.out_proj.bias,
                    A,A_b,None,None,self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
            elif self.bimamba_type=='v2':
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,)
                if not self.if_divide_out:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight,
                                   self.out_proj.bias)
                else:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight,
                                   self.out_proj.bias)
            else: #普通的扫描模式-->单向扫描
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x,z = xz.chunk(2,dim=1)
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seq_len])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seq_len)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seq_len).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seq_len).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out



