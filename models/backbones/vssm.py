'''
@File: vssm.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 25, 2024
@HomePage: https://github.com/YanJieWen
'''

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.blocks.norms import LayerNorm2d
from models.utils.misc import Permute
from models.blocks.patchmerge import PatchMerging2D
from models.mamba_zoo.visual_mamba import VSSBlock
from models.ssm_zoo.ss2d import SS2D

class VSSM(nn.Module):
    def __init__(self,
                 patch_size:int=4,
                 in_chans:int=3,
                 num_classes:int=1000,
                 depths:list=[2,2,9,2],
                 dims:list=[96,192,384,768],
                 #========================ssm_cfg
                 ssm_d_state:int=16,
                 ssm_ratio:float=2.0,
                 ssm_dt_rank:str='auto',
                 ssm_act_layer:str='silu',
                 ssm_conv:int=3,
                 ssm_conv_bias:bool=True,
                 ssm_drop_rate:float=0.0,
                 ssm_init:str='v0',
                 forward_type:str='v2',
                 #========================mlp_cfg
                 mlp_ratio:float=4.0,
                 mlp_act_layer:str='gelu',
                 mlp_drop_rate:float=0.0,
                 gmlp:bool=False,
                 # =========================
                 drop_path_rate=0.1,
                 patch_norm=True,
                 norm_layer="LN",  # "BN", "LN2D"
                 downsample_version: str = "v2",  # "v1", "v2", "v3"
                 patchembed_version: str = "v1",  # "v1", "v2"
                 use_checkpoint=False, #节约内存空间
                 # =========================
                 posembed=False,
                 imgsize=224,
                 _SS2D=SS2D,#SS2D
                 if_include_top:bool=True,):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ['bn','ln2d']) #默认为False
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.if_include_top = if_include_top
        if isinstance(dims,int):
            dims = [int(dims*2**x) for x in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,sum(depths))]
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS=dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        #定义norm_layer
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        #定义位置编码
        self.pos_embed = self._pos_embed(dims[0],patch_size,imgsize)#[1,embed_dim,p_h,p_w]
        #定义补丁编码,共计两种编码模式:V1:将图像率先下采样patch_size,v2:分两次连续卷积进行下采样4
        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm,
                                             norm_layer, channel_first=self.channel_first)
        _make_downsample  = dict(
            v1=PatchMerging2D, #需要注意是否为channel_first
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
        ).get(downsample_version, None)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer+1],
                norm_layer = norm_layer,
                channel_first=self.channel_first
            ) if (i_layer<self.num_layers-1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                _SS2D=_SS2D,
            ))

            if if_include_top:
                self.classifier = nn.Sequential(OrderedDict(
                    norm=norm_layer(self.num_features),  # B,H,W,C
                    permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
                    avgpool=nn.AdaptiveAvgPool2d(1),
                    flatten=nn.Flatten(1),
                    head=nn.Linear(self.num_features, num_classes),
                ))
            self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        nn.init.trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True,
                          norm_layer=nn.LayerNorm, channel_first=False):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True,
                             norm_layer=nn.LayerNorm, channel_first=False):
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1,0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank='auto',
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init='v0',
            forward_type='v2',
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # ==========================
            _SS2D=None
    ):
        #实现堆叠N个block得vssblock
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self,x:torch.Tensor):
        x = self.patch_embed(x)#b,h,w,c
        if self.pos_embed is not None:#是否可以考虑在此处引入相对位置编码
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x
    def load_pretrained_model(self,checkpoint):
        checkpoint_model = torch.load(checkpoint,map_location='cpu')['model']
        state_dict = self.state_dict()
        del_keys = [k for k,v in checkpoint_model.items() if 'classifier' in k]
        for k in del_keys:
            if k in checkpoint_model and checkpoint_model[k].shape!=state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        self.load_state_dict(checkpoint_model,strict=False)


# if __name__ == "__main__":
#     channel_first = True
#     # model = VSSM(
#     #     depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2,
#     #     patch_size=4, in_chans=3, num_classes=1000,
#     #     ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
#     #     ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
#     #     ssm_init="v0", forward_type="v0",
#     #     mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
#     #     patch_norm=True, norm_layer="ln",
#     #     downsample_version="v1", patchembed_version="v1",
#     #     use_checkpoint=False, posembed=False, imgsize=224,
#     # )
#     #
#     # #base
#     # model = VSSM(
#     #         depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6,
#     #         patch_size=4, in_chans=3, num_classes=1000,
#     #         ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
#     #         ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
#     #         ssm_init="v0", forward_type="v05_noz",
#     #         mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
#     #         patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
#     #         downsample_version="v3", patchembed_version="v2",
#     #         use_checkpoint=False, posembed=False, imgsize=224,
#     #     )
#
#     #small
#     model = VSSM(
#         depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3,
#         patch_size=4, in_chans=3, num_classes=1000,
#         ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
#         ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
#         ssm_init="v0", forward_type="v05_noz",
#         mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
#         patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
#         downsample_version="v3", patchembed_version="v2",
#         use_checkpoint=False, posembed=False, imgsize=224,
#     )
#     print(len([p for p in model.parameters() if p.requires_grad]))
#     model.cuda().train()
#
#
#     def bench(model):
#         import time
#         inp = torch.randn((4, 3, 224, 224)).cuda()
#         for _ in range(30):
#             model(inp)
#         torch.cuda.synchronize()
#         tim = time.time()
#         for _ in range(30):
#             model(inp)
#         torch.cuda.synchronize()
#         tim1 = time.time() - tim
#
#         for _ in range(30):
#             model(inp).sum().backward()
#         torch.cuda.synchronize()
#         tim = time.time()
#         for _ in range(30):
#             model(inp).sum().backward()
#         torch.cuda.synchronize()
#         tim2 = time.time() - tim
#
#         return tim1 / 30, tim2 / 30
#
#     print(bench(model))
#
#     breakpoint()
