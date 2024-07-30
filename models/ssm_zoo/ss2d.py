'''
@File: ss2d.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 26, 2024
@HomePage: https://github.com/YanJieWen
'''


import torch

import torch.nn as nn
from models.ssm_zoo.scan2d_factory.ss2d_ver01 import SS2Dv0
from models.ssm_zoo.scan2d_factory.ss2d_ver02 import SS2Dv2


class SS2D(nn.Module,SS2Dv0,SS2Dv2):
    def __init__(self,
                 d_model=96,
                 d_state=16,
                 ssm_ratio=2.0,
                 dt_rank='auto',
                 act_layer=nn.SiLU,
                 d_conv=3,
                 conv_bias=True,
                 dropout=0.0,
                 bias=False,
                 #dt init
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 initialize="v0",
                 forward_type="v2",
                 channel_first=False,**kwargs):
        nn.Module.__init__(self) #继程module的forward模块
        kwargs.update(d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ['v0','v0seq']:
            self.__initv0__(seq=('seq' in forward_type),**kwargs)
        else:
            self.__initv2__(**kwargs)