'''
@File: smoothingce.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 23, 2024
@HomePage: https://github.com/YanJieWen
'''


import torch
import torch.nn as nn

import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self,smoothing=0.1,reduction='mean'):
        super().__init__()
        assert smoothing<1.0
        self.smoothing = smoothing
        self.confidence =1.-smoothing
        self.reduction = reduction

    def forward(self,x,target):
        '''

        :param x:n,c
        :param target:n with cls_idx
        :return: data a value
        '''
        log_props = torch.log(torch.softmax(x,dim=-1))
        nll_loss = -log_props.gather(dim=-1,index=target.unsqueeze(1).to(dtype=torch.int64))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_props.mean(dim=-1)
        loss = self.confidence*nll_loss+self.smoothing*smooth_loss
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'{self.reduction} is not supported type')
