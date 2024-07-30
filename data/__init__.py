'''
@File: __init__.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 22, 2024
@HomePage: https://github.com/YanJieWen
'''


import importlib
from os import path as osp
import os

data_folder = osp.dirname(osp.abspath(__file__))
def create_datasets(data_name:str='cifar100',
                    data_type:str='CiFar100',
                    opt:dict=None):
    if opt is None:
        opt = {}
    data_filenames = [osp.splitext(osp.basename(v))[0] for _,_,k in os.walk(data_folder) for v in k if v.endswith(f'{data_name}.py')]
    assert len(data_filenames)!=0
    build_datas = [importlib.import_module(f'data.{data}') for data in data_filenames]
    for m in build_datas:
        data_builder = getattr(m,data_type,None)
        if data_builder is not None:
            break
    if data_builder is None:
        raise ValueError(f'Datasets {data_type} is not found.')
    data = data_builder(**opt)
    return data



# if __name__ == '__main__':
#     create_datasets()
