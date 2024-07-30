'''
@File: main.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 20, 2024
@HomePage: https://github.com/YanJieWen
'''

import os
import json
import random
import numpy as np
import copy
import torch

import time

from models import create_backbone #创建模型
from data import create_datasets #创建数据构造器
from engines import create_trainer,create_evaler #创建训练驱动以及评估驱动

from tools.get_gflops import cal_params
from tools.parse_yaml import parse_cfg

import argparse


def main(args):
    opt = parse_cfg(args.opt,is_train=args.train)
    #random seed
    seed =  opt.get('manual_seed')
    if seed is None:
           seed = random.randint(1, 10000)
           opt['manual_seed'] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #part1: 数据申明
    assert 'datasets' in opt.keys(),'datasets is not in the cfg options'
    data_cfg = opt['datasets']
    print(f'Creating data: {data_cfg["data_type"]}')
    basic_data = {'data_name':data_cfg['name'],'data_type':data_cfg['data_type']}
    basic_data['opt'] = data_cfg['train']
    train_data = create_datasets(**basic_data)
    basic_data['opt'] = data_cfg['val']
    val_data = create_datasets(**basic_data)
    train_loader = torch.utils.data.DataLoader(train_data,args.batch_size,shuffle=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data,args.batch_size,shuffle=False,num_workers=4)

    #part2: 创建模型
    assert 'backbone' in opt.keys(),'backbone is not in the cfg options'
    print(f'Creating model: {opt["model_type"]}')
    backbone_cfg = opt['backbone']
    model_cfg = {'bb_name':opt['name'],'model_type':opt['model_type'],
                 'opt':backbone_cfg['model_cfg'],'pretrained':backbone_cfg['ckpt_path']}
    model = create_backbone(**model_cfg)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cal_params(model,224)
    # x = torch.rand(4,3,224,224).to('cuda')
    # print(model(x).shape)

    #part3: 训练流程
    model.to(device)
    print(f'Creating engines: {opt["train"]["name"]}')
    train_cfg = opt['train']
    optimizer_cfg = train_cfg['optimizer']
    optim_name = optimizer_cfg.pop('name')
    lrsc_cfg = train_cfg['lr_schedular']
    lrsc_name = lrsc_cfg.pop('name')
    loss_cfg = train_cfg['criterion']
    loss_name = loss_cfg.pop('name')

    trainer_cfg = {'model':model,'optimizer':optim_name,'optim_cfg':optimizer_cfg,'lr_schedular':lrsc_name,
                   'sc_cfg':lrsc_cfg,'criterion':loss_name,'loss_cfg':loss_cfg,'device':device,'resume':train_cfg['resume']}
    build_trainer_cfg = {'engine_name':train_cfg['name'],'engine_type':train_cfg['engine_type'],
                         'opt':trainer_cfg}
    trainer = create_trainer(**build_trainer_cfg)
    print(f'Creating engines: {opt["val"]["name"]}')
    val_cfg = opt['val']
    metric_cfg = val_cfg['metric']
    evaler_cfg = {'metric':metric_cfg['name'],'device':device}
    build_evaler_cfg = {'engine_name':val_cfg['name'],'engine_type':val_cfg['engine_type'],
                        'opt':evaler_cfg}
    evaer = create_evaler(**build_evaler_cfg)

    num_epochs,if_continue_inf,warmup = args.num_epochs,args.if_continue_inf,args.warmup
    start_epoch = trainer.start_epoch
    max_accuracy = 0.0
    start_time = time.time()
    for epoch in range(start_epoch,num_epochs):
        mloss,train_stats = trainer(train_loader,epoch,warmup,if_continue_inf)
        trainer.lr_schedular.step(epoch)
        test_stats = evaer(trainer.model,val_loader)
        print(f"Accuracy of the network on the {len(val_data)} test images: {test_stats['acc1']:.1f}%")
        save_files = {
            'model': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'lr_scheduler': trainer.lr_schedular.state_dict(),
            'epoch': epoch}
        if max_accuracy<test_stats["acc1"]:
            max_accuracy=test_stats['acc1']
            if not os.path.exists(args.save_weights):
                os.makedirs(args.save_weights)
            torch.save(save_files,f'{args.save_weights}/{opt["name"]}_{epoch}@%.2f.pth'%max_accuracy)
        print(f'Max accuracy: {max_accuracy:.2f}%')
        # log_stats = [**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters]
        log_stats = [f'train_{k}:{v}' for k,v in train_stats.items()]+[f'val_{k}:{v}' for k,v in test_stats.items()]+[f'{mloss.item()}']+[f'n_parameters:{n_parameters}']
        with open('results_info.txt','a') as w:
            txt = "epoch:{} {}".format(epoch, '  '.join(log_stats))
            w.write(txt+'\n')
    total_time = time.time() - start_time
    import datetime
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./yamls/Vmamba/vmamba-base.yml', help='all cfg options')
    parser.add_argument('--train', type=bool, default=True, help='train mode')
    parser.add_argument('--batch_size', type=int, default=12, help='number of samples')
    parser.add_argument('--warmup', type=bool, default=True, help='if the first epoch warmup')
    parser.add_argument('--if_continue_inf', type=bool, default=False, help='if loss inf zero_grad')
    parser.add_argument('--num_epochs', type=int, default=12, help='number of training')
    parser.add_argument('--save_weights', type=str, default='./save_weights/', help='weights saving dir')
    args = parser.parse_args()
    main(args)



