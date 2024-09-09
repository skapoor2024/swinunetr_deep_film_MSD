import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.epsilon = 1e-7

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(1,-1)
        target = target.contiguous().view(1,-1)

        #print(predict.shape)
        #print(target.shape)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

        return dice_loss_avg

    """
    Editing the loss function. Since the given input prediection and target are without passed 
    with batch and template . Therefore not averaging where the target values are -1.
    """

    # def forward(self, predict, target):
    #     assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    #     predict = predict.contiguous().view(-1)
    #     target = target.contiguous().view(-1)

    #     #print(predict.shape)
    #     #print(target.shape)

    #     num = torch.sum(torch.mul(predict, target), dim=0)
    #     den = torch.sum(predict, dim=0) + torch.sum(target, dim=0) + self.smooth

    #     dice_score = 2*num / den
    #     dice_loss = 1 - dice_score

    #     #dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

    #     return dice_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, name, TEMPLATE):
        
        total_loss = []
        predict = F.sigmoid(predict)
        B = predict.shape[0]

        for b in range(B):
            dataset_index = int(name[b][0:2])
            if dataset_index == 10:
                template_key = name[b][0:2] + '_' + name[b][17:19]
            elif dataset_index == 1:
                if int(name[b][-2:]) >= 60:
                    template_key = '01_2'
                else:
                    template_key = '01'
            else:
                template_key = name[b][0:2]
            organ_list = TEMPLATE[template_key]

            #print(template_key)
            #print(organ_list)

            # for organ in organ_list:
            #     #print(predict[b, organ-1].shape)
            #     #print(target[b, organ-1].shape)
            #     dice_loss = self.dice(predict[b, organ-1], target[b, organ-1])
            #     total_loss.append(dice_loss)

            dice_loss = self.dice(predict[b,organ_list],target[b,organ_list])
            total_loss.append(dice_loss)
            
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/B

        

class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, name, TEMPLATE):
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            dataset_index = int(name[b][0:2])
            if dataset_index == 10:
                template_key = name[b][0:2] + '_' + name[b][17:19]
            elif dataset_index == 1:
                if int(name[b][-2:]) >= 60:
                    template_key = '01_2'
                else:
                    template_key = '01'
            else:
                template_key = name[b][0:2]
            organ_list = TEMPLATE[template_key]
            #print(template_key)
            #print(organ_list)
            # for organ in organ_list:
            #     #print(predict[b, organ-1].shape)
            #     #print(target[b, organ-1].shape)
            #     ce_loss = self.criterion(predict[b, organ-1], target[b, organ-1])
            #     total_loss.append(ce_loss)

            ce_loss = self.criterion(predict[b,organ_list],target[b,organ_list])
            total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)

        # print(name, total_loss, total_loss.sum()/total_loss.shape[0])

        return total_loss.sum()/B
