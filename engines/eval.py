'''
@File: eval.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 23, 2024
@HomePage: https://github.com/YanJieWen
'''

from functools import partial

import torch

from engines.utils.metrics_factory.accuracy import accuracy
import engines.mm_utils as mmu

class Evaluator(object):
    def __init__(self,metric='topk',device='cuda:0'):
        metric_lower = metric.lower()
        if metric_lower == 'topk':
            self.metric = partial(accuracy, topk=(1,5))
        else:
            raise ValueError(f'only {metric} is supported')
        self.device = device

    @torch.no_grad()
    def __call__(self, model,data_loader):
        model.to(self.device)
        model.eval()
        metric_logger = mmu.MetricLogger(delimiter="  ")
        header = 'Test:'
        for images, target in metric_logger.log_every(data_loader, 10, header):
            images = images.to(self.device)
            target = target.to(self.device)
            output = model(images)
            acc1, acc5 = self.metric(output=output, target=target)
            batch_size = images.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



