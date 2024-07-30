'''
@File: __init__.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 20, 2024
@HomePage: https://github.com/YanJieWen
'''

import importlib
from os import path as osp
import os

backbone_folder = osp.dirname(osp.abspath(__file__))

def create_backbone(bb_name:str='vim',
                    model_type:str='VisionMamba',
                    opt:dict=None,
                    pretrained:str=None):
    if opt is None:
        opt = {}
    model_filenames = [
        osp.splitext(osp.basename(v))[0] for _,_,k in os.walk(backbone_folder) for v in k
        if v.endswith(f'{bb_name}.py')
    ]
    assert len(model_filenames)!=0
    _model_modules = [importlib.import_module(f'models.backbones.{file_name}') for file_name in model_filenames]
    for m in _model_modules:
        model_cls = getattr(m,model_type,None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')
    model = model_cls(**opt)
    if pretrained is not None:
        model.load_pretrained_model(pretrained)
    return model




