'''
@File: crossentropy.py
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


class CrossEntropy(nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self,pred,gt):
        '''

        :param pred: n,c
        :param gt: n/(n,c)
        :return:
        '''
        log_pred = F.log_softmax(pred,dim=-1)
        if len(gt.shape)==1:
            nll_loss = -log_pred.gather(dim=-1,index=gt.unsqueeze(1))
        elif len(gt.shape)==2:
            nll_loss = torch.sum(-gt*log_pred,dim=-1)
        else:
            raise ValueError(f'gt size {gt.shape} is not supported')
        if self.reduction == 'sum':
            return nll_loss.sum()
        elif self.reduction == 'mean':
            return nll_loss.mean()
        elif self.reduction == 'none':
            return nll_loss
        else:
            raise ValueError(f'{self.reduction} is not supported type')