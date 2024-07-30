'''
@File: cifar100.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 7月 22, 2024
@HomePage: https://github.com/YanJieWen
'''

import torchvision
import torchvision.transforms as tss
import numpy as np
from torch.utils.data import Dataset
import os



class CiFar100(Dataset):
    def __init__(self,root='./datasets/cifar100',mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],
                 gt_size=224,data_type='Train'):
        super().__init__()
        self.mean = mean if not None else None
        self.std = std if not None else None
        self.root = root
        if not os.path.exists(root):
            os.makedirs(root)
        self.gt_size = gt_size
        if data_type=='Train':
            transform = tss.Compose([tss.RandomResizedCrop(gt_size),tss.RandomHorizontalFlip(),tss.ColorJitter(),
        tss.ToTensor(),tss.Normalize(mean,std)])
        else:
            transform = tss.Compose([tss.Resize(256),tss.CenterCrop(gt_size),tss.ToTensor(),tss.Normalize(mean,std)])
        self.cifar100_data = torchvision.datasets.CIFAR100(root=root,train= data_type=='Train',download=True,transform=transform)

    def __len__(self):
        return len(self.cifar100_data)

    def __getitem__(self, idx):
        img,cls = self.cifar100_data[idx]
        return img,cls

    # @staticmethod
    # def collate_fn(batch):
    #     return tuple(zip(*batch))




# if __name__ == '__main__':
#     import torch
#     from PIL import Image
#     import matplotlib.pyplot as plt
#
#     cifar = CiFar100(data_type='Test')
#     trainloader = torch.utils.data.DataLoader(cifar,batch_size=4,shuffle=True,num_workers=4,collate_fn=None)
#     batch_img,batch_label = next(iter(trainloader))
#     for img in batch_img:
#         img = img*torch.as_tensor(np.array(cifar.std))[:,None,None]+torch.as_tensor(np.array(cifar.mean))[:,None,None]
#         img = tss.ToPILImage()(img)
#         img = tss.Resize(1248)(img)
#         plt.imshow(img)
#         plt.show()
