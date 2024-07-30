'''
@File: visual_mamba.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 26, 2024
@HomePage: https://github.com/YanJieWen
'''


import torch
import torch.nn as nn

from typing import Optional,Any
from models.blocks.drop_path import DropPath
from models.blocks.mlp import Mlp,gMlp

from torch.utils import checkpoint

class VSSBlock(nn.Module):
    def __init__(self,
                 hidden_dim:int=0,
                 drop_path:float=0.,
                 norm_layer:nn.Module=nn.LayerNorm,
                 channel_first:bool=False,
                 ssm_d_state: int = 16,
                 ssm_ratio=2.0,
                 ssm_dt_rank: Any = "auto",
                 ssm_act_layer=nn.SiLU,
                 ssm_conv: int = 3,
                 ssm_conv_bias=True,
                 ssm_drop_rate: float = 0,
                 ssm_init="v0",
                 forward_type="v2",
                 mlp_ratio=4.0,
                 mlp_act_layer=nn.GELU,
                 mlp_drop_rate: float = 0.0,
                 gmlp=False,
                 use_checkpoint: bool = False,
                 post_norm: bool = False,
                 _SS2D=None,):
        super().__init__()
        self.ssm_branch = ssm_ratio>0
        self.mlp_branch =mlp_ratio>0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op =_SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                drop_out=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first
            )
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim*mlp_ratio)
            #瓶颈层
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim,
                           act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)


    def _forward(self,input):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x+self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self,input):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward,input)
        else:
            return self._forward(input)

