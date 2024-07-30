'''
@File: __init__.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 23, 2024
@HomePage: https://github.com/YanJieWen
'''

import importlib
from os import path as osp
import os

engine_folder = osp.dirname(osp.abspath(__file__))

def create_trainer(
        engine_name:str = 'train',
        engine_type:str='Trainer',
        opt:dict=None):
    if opt is None:
        opt = {}
    engine_filenames = [osp.splitext(osp.basename(v))[0] for _,_,k in os.walk(engine_folder)
                        for v in k if v.endswith(f'{engine_name}.py')]
    assert len(engine_filenames) != 0
    build_engines = [importlib.import_module(f'engines.{engine}') for engine in engine_filenames]
    for m in build_engines:
        engine_builder = getattr(m,engine_type,None)
        if engine_builder is not None:
            break
    if engine_builder is None:
        raise ValueError(f'Enginer {engine_type} is not found.')
    engine = engine_builder(**opt)
    return engine


def create_evaler(engine_name:str = 'eval',
        engine_type:str='Evaluator',
        opt:dict=None):
    if opt is None:
        opt = {}
    engine_filenames = [osp.splitext(osp.basename(v))[0] for _,_,k in os.walk(engine_folder)
                        for v in k if v.endswith(f'{engine_name}.py')]
    assert len(engine_filenames) != 0
    build_engines = [importlib.import_module(f'engines.{engine}') for engine in engine_filenames]
    for m in build_engines:
        engine_builder = getattr(m,engine_type,None)
        if engine_builder is not None:
            break
    if engine_builder is None:
        raise ValueError(f'Enginer {engine_type} is not found.')
    engine = engine_builder(**opt)
    return engine
    return None


