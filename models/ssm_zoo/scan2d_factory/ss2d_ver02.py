'''
@File: ss2d_ver02.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 29, 2024
@HomePage: https://github.com/YanJieWen
'''


import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from functools import partial
from models.mamba_zoo.mambaint import mamba_init
from models.ssm_zoo.ssm_methods import selective_scan_fn
from models.blocks.patchmerge import Linear2d
from models.blocks.norms import LayerNorm2d
from models.utils.misc import Permute
from models.ssm_zoo.crossm_methods import cross_scan_fn,cross_merge_fn

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class SS2Dv2:
    def __initv2__(self,
                   d_model=96,
                   d_state=16,
                   ssm_ratio=2.0,
                   dt_rank='auto',
                   act_layer=nn.SiLU,
                   d_conv=3,
                   conv_bias=True,
                   dropout=0.0,
                   bias=False,
                   dt_min=0.001,
                   dt_max=0.1,
                   dt_init='random',
                   dt_scale=1.0,
                   dt_init_floor=1e-4,
                   initialize='v0',
                   forward_type='v2',
                   channel_first=False,
                   **kwargs):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2 #在此处继承module的forward
        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)


        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
                        scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner))) #torch.rand

    def forward_corev2(self,x,force_fp32=False,no_einsum=False,scan_mode = "cross2d",):
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) \
            if isinstance(scan_mode, str) else scan_mode
        assert isinstance(_scan_mode,int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W
        selective_scan = partial(selective_scan_fn,delta_softplus=delta_softplus,return_last_state=False)

        if _scan_mode==-1:#2d级联扫描
            x_proj_bias = getattr(self,'x_proj_bias',None)
            As = -self.A_logs.to(torch.float32).exp().view(4,-1,N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1,
                                                                                                 2).contiguous()
            y_row = self.scan_rowcol(x,proj_weight=self.x_proj_weight.view(4,-1,D)[:2].contiguous(),
                                     proj_bias=(x_proj_bias.view(4,D)[:2].contiguous() if x_proj_bias is not None else None),
                                     dt_weight=self.dt_projs_weight.view(4,D,-1)[:2].contiguous(),
                                     dt_bias=(self.dt_projs_bias.view(4,-1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                                     _As=As[:2].contiguous().view(-1,N),
                                     _Ds=self.Ds.view(4,-1)[:2].contiguous().view(-1),
                                     R=R,N=N,width=True,no_einsum=no_einsum,
                                     force_fp32=force_fp32,selective_scan=selective_scan).view(B,H,2,-1,W).sum(dim=2).permute(0,2,1,3)#b,c,h,w
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous()
            y_col = self.scan_rowcol(y_row,proj_weight=self.x_proj_weight.view(4,-1,D)[2:].contiguous(),
                                     proj_bias=(x_proj_bias.view(4,D)[2:].contiguous() if x_proj_bias is not None else None),
                                     dt_weight=self.dt_projs_weight.view(4,D,-1)[2:].contiguous().to(y_row.dtype),
                                     dt_bias=(self.dt_projs_bias.view(4,-1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                                     _As = As[2:].contiguous().view(-1, N),
                                     _Ds = self.Ds.view(4, -1)[2:].contiguous().view(-1),
                                     width=False,R=R,N=N,no_einsum=no_einsum,
                                     force_fp32=force_fp32,selective_scan=selective_scan).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            #reverse xs->b,k,c,h,w
            xs = cross_scan_fn(x,in_channel_first=True,out_channel_first=True,scans=_scan_mode,one_by_one=False)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum('b k d l, k c d->b k c l',xs,self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl+x_proj_bias.view(1,K,-1,1)
                dts,Bs,Cs = torch.split(x_dbl,[R,N,N],dim=2)
                if hasattr(self,"dt_projs_weight"):
                    dts = torch.einsum('b k r l,k d r->b k d l',dts,self.dt_projs_weight)
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L).repeat(1,D,1,1)#(B,KD,N,L)
            Cs = Cs.contiguous().view(B, K, N, L).repeat(1,D,1,1)#(B,KD,N,L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)
            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
            ys = selective_scan(xs,dts,As,Bs,Cs,Ds,delta_bias=delta_bias,z=None).view(B, K, -1, H, W)
            y = cross_merge_fn(ys,in_channel_first=True,out_channel_first=True,scans=_scan_mode,one_by_one=False)

        y = y.view(B,-1,H,W)
        if not channel_first:
            y = y.view(B,-1,H*W).transpose(dim0=1,dim1=2).contiguous().view(B,H,W,-1)
        y = out_norm(y)
        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)#ss2d
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):#bool&forward_version
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

    @staticmethod
    def scan_rowcol(x,proj_weight,proj_bias,dt_weight,dt_bias,R,N,_As,_Ds,
                    width=True,no_einsum=True,force_fp32=True,selective_scan=None):
        '''
        并行化扫描行和列
        :param x: b,d,h,w
        :param proj_weight: (d_r+d_s*2),d_in
        :param proj_bias: d_in
        :param dt_weight: k,d_in,d_r
        :param dt_bias: d_in
        :param R: number of rank
        :param N: number of dim_states
        :param _As: -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
        :param _Ds:k*d_in
        :param width:bool:是否为行扫描
        :param no_einsum:是否采用矩阵规约
        :param force_fp32:bool
        :param selective_scan:fn
        :return:bh,k,d_in,w
        '''
        # As =
        # the same as Ds
        XB,XD,XH,XW = x.shape
        if width:
            _B,_D,_L =XB*XH,XD,XW
            xs = x.permute(0,2,1,3).contiguous()#b,h,c,w
        else:
            _B, _D, _L = XB * XW, XD, XH #scan height,并行化扫描宽和高
            xs = x.permute(0, 3, 1, 2).contiguous() #b,w,c,h
        xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W) width inverse
        if no_einsum:
            x_dbl = F.conv1d(xs.view(_B,-1,_L),proj_weight.view(-1,_D,1),
                             bias=(proj_bias.view(-1) if proj_bias is not None else None),groups=2)
            dts,Bs,Cs = torch.split(x_dbl.view(_B,2,-1,_L),[R,N,N],dim=2)
            dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
            if proj_bias is not None:
                x_dbl = x_dbl + proj_bias.view(1, 2, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)
        xs = xs.view(_B, -1, _L)#[bh,2d_in,w]
        dts = dts.contiguous().view(_B, -1, _L)#[bh,2d_in,w]
        As = _As.view(-1, N).to(torch.float)#2d_in,d_s
        Bs = Bs.contiguous().view(_B, 2, N, _L)#bh,2,d_n,w
        Cs = Cs.contiguous().view(_B, 2, N, _L)#bh,2,d_n,w
        Ds = _Ds.view(-1)#2d_in
        delta_bias = dt_bias.view(-1).to(torch.float)#2d_in
        if force_fp32:
            xs = xs.to(torch.float)
        dts = dts.to(xs.dtype)
        Bs = Bs.to(xs.dtype)
        Cs = Cs.to(xs.dtype)
        Bs = Bs.repeat(1,_D,1,1)
        Cs = Cs.repeat(1,_D,1,1)
        ys = selective_scan(xs,dts,As,Bs,Cs,Ds,delta_bias=delta_bias,z=None).view(_B,2,-1,_L)#bh,2,d_in,w
        assert ys.dtype==torch.float
        return ys


