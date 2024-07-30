'''
@File: train.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 23, 2024
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn
import torch.optim as optim

import math

from functools import partial

from .utils.optimizers_factory import adabelif,nadamw,radam
from .utils.lrsc_factory import cosine_lr,multistep_lr,poly_lr,tanh_lr
from .utils.criterion_factory import crossentropy,smoothingce
import engines.mm_utils as mmu



import sys


class Trainer(object):
    def __init__(self,
                 model=None,
                 optimizer:str='sgd',
                 optim_cfg:dict=None,
                 lr_schedular:str='linear',
                 sc_cfg:dict=None,
                 criterion:str='bce',
                 loss_cfg:dict=None,
                 device:str='cuda:0',
                 resume:str=None):
        self.device = device
        self.model = model.to(device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        #create optimizer
        if optim_cfg is None:
            optim_cfg = {}
        opt_lower = optimizer.lower()
        if opt_lower == 'sgd':
            self.optimizer = partial(optim.SGD,nesterov=True,**optim_cfg)
        elif opt_lower == 'adam':
            self.optimizer = partial(optim.Adam,**optim_cfg)
        elif opt_lower == 'adamw':
            self.optimizer = partial(optim.AdamW,**optim_cfg)
        elif opt_lower == 'nadam':
            self.optimizer = partial(optim.NAdam,**optim_cfg)
        elif opt_lower == 'adamax':
            self.optimizer = partial(optim.Adamax,**optim_cfg)
        elif opt_lower == 'adadelta':
            self.optimizer = partial(optim.Adadelta,**optim_cfg)
        elif opt_lower =='adagrad':
            self.optimizer = partial(optim.Adagrad,**optim_cfg)
        elif opt_lower == 'rmsprop':
            self.optimizer = partial(optim.RMSprop,**optim_cfg)
        elif opt_lower == 'adabelif':
            self.optimizer = partial(adabelif.AdaBelief,rectify=True,**optim_cfg)
        elif opt_lower == 'nadamw':
            self.optimizer = partial(nadamw.NAdamW,**optim_cfg)
        elif opt_lower == 'radam':
            self.optimizer = partial(radam.RAdam,**optim_cfg)
        else:
            raise ValueError(f'optim {opt_lower} is not supported')

        #create lr_schedular
        if sc_cfg is None:
            sc_cfg = {}
        sc_lower = lr_schedular.lower()
        if sc_lower == 'linear':
            self.lr_schedular = partial(optim.lr_scheduler.StepLR,**sc_cfg)
        elif sc_lower == 'cosine':
            self.lr_schedular = partial(cosine_lr.CosineLRScheduler,**sc_cfg)
        elif sc_lower == 'multistep':
            self.lr_schedular = partial(multistep_lr.MultiStepLRScheduler,**sc_cfg)
        elif sc_lower == 'poly':
            self.lr_schedular = partial(poly_lr.PolyLRScheduler,**sc_cfg)
        elif sc_lower == 'tanh':
            self.lr_schedular = partial(tanh_lr.TanhLRScheduler,**sc_cfg)
        else:
            raise ValueError(f'lr_sc {sc_lower} is not supported')
        self.optimizer = self.optimizer(params=params)
        self.lr_schedular = self.lr_schedular(optimizer=self.optimizer)
        self.start_epoch = 0
        if resume != " ":
            chckpoint = torch.load(resume,map_location='cpu')
            self.model.load_state_dict(chckpoint['model'] if 'model' in chckpoint.keys() else chckpoint)
            self.optimizer.load_state_dict(chckpoint['optimizer'])
            self.lr_schedular.load_state_dict(chckpoint['lr_scheduler'])
            self.start_epoch = chckpoint['epoch']+1 if 'epoch' in chckpoint.keys() else 0
        print(f'Training process from epoch {int(self.start_epoch)}')
        #create criterion
        if loss_cfg is None:
            loss_cfg = {}
        cre_lower = criterion.lower()
        if cre_lower == 'bce': #pred:(n,c)/c-->gt:(n,c)/c
            self.creterion = nn.BCEWithLogitsLoss(**loss_cfg)
        elif cre_lower == 'smooth':#pred:(n,c)-->gt:c
            self.creterion = smoothingce.LabelSmoothingCrossEntropy(**loss_cfg)
        elif cre_lower == 'ce': #pred:(n,c)-->gt:(n,c)/c
            self.creterion = crossentropy.CrossEntropy(**loss_cfg)
        else:
            raise ValueError(f'loss fn {cre_lower} is not supported')


    def __call__(self,data_loader,epoch,warmup=True,if_continue_inf=False):
        '''
        run train_one_epoch
        :param data_loader:Dataset
        :param epoch:int
        :param warmup:bool
        :param if_continue_inf:bool
        :return:
        '''
        self.model.train()
        metric_logger = mmu.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr',mmu.SmoothedValue(window_size=1,fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 50

        lr_schedular = None
        if epoch==0 and warmup is True:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)
            lr_schedular = mmu.warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)
        mloss = torch.zeros(1).to(self.device)
        for i,(samples,targets) in enumerate(metric_logger.log_every(data_loader,print_freq,header)):
            samples = samples.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(samples)
            loss = self.creterion(outputs,targets)
            if torch.is_tensor(loss):
                loss_value = loss.sum().item()
            else:
                loss_value = loss.item()
            mloss = (mloss*i+loss_value)/(i+1)
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                if if_continue_inf:
                    self.optimizer.zero_grad()
                    continue
                else:
                    sys.exit(1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if lr_schedular is not None:
                lr_schedular.step()
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        print("Averaged stats:", metric_logger)
        train_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        return mloss,train_state
