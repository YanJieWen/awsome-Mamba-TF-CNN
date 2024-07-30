'''
@File: parse_yaml.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 22, 2024
@HomePage: https://github.com/YanJieWen
'''


import yaml
from collections import OrderedDict

from os import path as osp


def orderd_yaml():
    '''
    https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/utils/options.py#L31
    :return:
    '''
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def parse_cfg(cfg_path,is_train=True):
    with open(cfg_path,mode='r') as f:
        Loader,_ = orderd_yaml()
        opt = yaml.load(f,Loader=Loader)
    opt['is_train'] = is_train
    return opt





# if __name__ == '__main__':
#     cfg_path = '../yamls/vims/demo.yml'
#     cfg = parse_cfg(cfg_path)
#     print(cfg)
