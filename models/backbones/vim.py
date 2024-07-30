'''
@File: vim.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 20, 2024
@HomePage: https://github.com/YanJieWen
'''

from functools import partial


import torch
import torch.nn as nn
import math


#模块导入
from ..mamba_zoo.bi_mamba import Mamba
from ..blocks.patch_embed import PatchEmbed
from ..blocks.drop_path import DropPath
from ..blocks.norms import RMSNorm,rms_norm_fn,layer_norm_fn


class Block(nn.Module):
    def __init__(self,dim,mixer_cls,norm_cls=nn.LayerNorm,fused_add_norm=False,drop_path=0.):
        super().__init__()
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.drop_path =DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm = norm_cls(dim)
        self.fused_add_norm = fused_add_norm
        if fused_add_norm:
            assert RMSNorm is not None and isinstance(self.norm,(nn.LayerNorm,RMSNorm)), 'Only Layernorm and RMSNorm are supported for fused_add_norm.'
    def forward(self,
                hidden_states: torch.Tensor = None,
                residual: torch.Tensor = None
                ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual+self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else: #使用fused_add_norm模式
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm,RMSNorm) else layer_norm_fn
            if residual is None:#用于处理第一个block为空
                hidden_states,residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    eps=self.norm.eps
                )
            else:
                hidden_states,residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    eps=self.norm.eps
                )
        hidden_states = self.mixer(hidden_states)
        return hidden_states,residual

def create_block(d_model,
                 ssm_cfg=None,
                 norm_epsilon=1e-5,
                 drop_path=0.,
                 rms_norm=False,
                 fused_add_norm=False,
                 layer_idx=None,
                 if_bimamba=False,
                 bimamba_type='none',
                 if_divide_out=False):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba,layer_idx=layer_idx,bimamba_type=bimamba_type,if_divide_out=if_divide_out,**ssm_cfg)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    block = Block(d_model,mixer_cls,norm_cls=norm_cls,drop_path=drop_path,fused_add_norm=fused_add_norm)
    block.layer_idx = layer_idx
    return block



class VisionMamba(nn.Module):
    '''
    https://github.com/hustvl/Vim
    '''
    def __init__(self,
                 img_size:int = 224,
                 patch_size:int = 16,
                 stride: int = 16,
                 channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 192,
                 drop_rate: float = 0.,
                 depth: int = 24,
                 drop_path_rate: float = 0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 fused_add_norm: bool = False,
                 if_bidirectional: bool = False,
                 final_pool_type: str='none',
                 if_abs_pos_embed: bool = False,
                 if_bimamba: bool = False,
                 bimamba_type: str = 'none',
                 if_cls_token: bool = False,
                 if_divide_out: bool = False,
                 ssm_cfg: dict = None,
                 use_middle_cls_token: bool = False):
        super().__init__()
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0

        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim# num_features for consistency with other models

        self.patch_embed = PatchEmbed(img_size=img_size,patch_size=patch_size,stride=stride,in_chans=channels,embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.ones(1,1,self.embed_dim))
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.ones(1,num_patches+self.num_tokens,self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()

        dpr =[x.item() for x in torch.linspace(0,drop_path_rate,depth)]
        inter_dpr = [0.0]+dpr
        #用于残差连接
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate>0. else nn.Identity()
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path = inter_dpr[i],
                if_divide_out=if_divide_out,
            )
            for i in range(depth)
        ])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim,eps=norm_epsilon)
        #origin_init
        self.patch_embed.apply(self.segm_init_weights)
        self.head.apply(self.segm_init_weights)
        if if_abs_pos_embed:
            nn.init.trunc_normal_(self.pos_embed,std=.02)
        if if_cls_token:
            nn.init.trunc_normal_(self.cls_token,std=.02)
        #kernel mamba init
        self.apply(partial(self._init_weights,n_layer=depth,))

    def segm_init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _init_weights(self,
            module,
            n_layer,
            initializer_range=0.02,  # Now only used for embedding layer.
            rescale_prenorm_residual=True,
            n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


    def forward_features(self,x):
        x = self.patch_embed(x)#b,m,c
        B,M,_ = x.shape
        if self.if_cls_token:
            cls_token = self.cls_token.expand(B,-1,-1)#b,1,d
            token_position = M//2
            x = torch.cat((x[:,:token_position,:],cls_token,x[:,token_position:,:]),dim=1)
            M = x.shape[1]#M+1
        if self.if_abs_pos_embed:
            x = x+self.pos_embed
            x = self.pos_drop(x)

        #mamba implement
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:
                hidden_states,residual = layer(hidden_states,residual)
        else:#双向
            for i in range(len(self.layers)//2):
                hidden_states_f,residual_f = self.layers[i*2](hidden_states,residual)
                hidden_states_b,residual_b = self.layers[i*2+1](hidden_states.flip([1]),None if residual==None else residual.flip([1]))
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual =residual,
                prenorm=False
            )
        if self.if_cls_token:
            return hidden_states[:,token_position,:]

        if self.final_pool_type == 'none':
            return hidden_states[:,-1,:]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self,x,return_features=False):
        x = self.forward_features(x)
        if return_features:
            return x
        x = self.head(x)
        if self.final_pool_type == 'max':#取出概率最大的
            x = x.max(dim=1)[0]
        return x

    def load_pretrained_model(self,checkpoint):
        checkpoint_model = torch.load(checkpoint,map_location='cpu')['model']
        state_dict = self.state_dict()
        for k in ['head.weight','head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape!=state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        self.load_state_dict(checkpoint_model,strict=False)
        # if finetune:
        #     pos_embed_checkpoint = checkpoint_model['pos_embed']
        #     embedding_size = pos_embed_checkpoint.shape[-1]
        #     num_patches = self.patch_embed.num_patches
        #     num_extra_tokens = self.pos_embed.shape[-2] - num_patches
        #     orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        #     new_size = int(num_patches ** 0.5)
        #     extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        #     pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        #     pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        #     pos_tokens = torch.nn.functional.interpolate(
        #         pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        #     pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        #     new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        #     checkpoint_model['pos_embed'] = new_pos_embed
        #     self.load_state_dict(checkpoint_model, strict=False)

        # head = self.head
        # if self.num_classes!=num_classes:
        #     self.head = nn.Linear(self.num_features,num_classes)
        # pth = torch.load(checkpoint)
        # pth = pth['model'] if 'model' in pth.keys() else pth
        # expt_key,miss_key = self.load_state_dict(pth,strict=False)
        # assert len(expt_key)==0,f'except keys-->{expt_key}'
        # assert len(miss_key)==0,f'missing keys-->{miss_key}'
        # self.head = head


#
# if __name__ == '__main__':
#     model = VisionMamba()
