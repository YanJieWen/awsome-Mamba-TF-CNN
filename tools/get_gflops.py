'''
@File: get_gflops.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 19, 2024
@HomePage: https://github.com/YanJieWen
'''


import thop
from thop import clever_format

import torch

def cal_params(model,img_size=224):
    '''
    计算模型的GFLOPS以及parameters
    :param model: nn.Module
    :param img_size: int:224
    :return: None
    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    inp = torch.rand(1,3,img_size,img_size).to(device)
    model = model.to(device)
    flops,params = thop.profile(model,inputs=(inp,))
    flops,params = clever_format([flops,params],'%.3f')
    print('='*20,f'FLOPS-->{flops},Params-->{params}','='*20)
