# train.py
import torch
from torch.utils.data import DataLoader
from torch.nn import L1Loss, MSELoss
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
from utils import get_gt

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def sparsity(arr, batch_size, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss

def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]  # f_t
    arr2[-1] = arr[-1]  # f_t-1

    loss = torch.sum((arr2 - arr) ** 2)
    return lamda1 * loss

def l1_penalty(var):
    return torch.mean(torch.norm(var, dim=0))

class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)

class SigmoidCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(-torch.abs(x))
        return torch.abs(torch.mean(-x * target + torch.clamp(x, min=0) + torch.log(tmp)))

class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin, normal_weight, abnormal_weight):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.normal_weight = normal_weight
        self.abnormal_weight = abnormal_weight
        if normal_weight == 1 and abnormal_weight == 1:
            self.criterion = torch.nn.BCELoss()
        else:
            self.criterion = None

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)  # shape=[64,]
        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()  # shape=[64,]
        label = label.cuda()

        if self.criterion:
            loss_cls = self.criterion(score, label)
        else:
            weight = self.abnormal_weight * label + self.normal_weight * (1 - label)
            self.criterion = torch.nn.BCELoss(weight)
            loss_cls = self.criterion(score, label)

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm
        return loss_total

def train(nloader, aloader, model, args, optimizer, viz, device):
    with torch.set_grad_enabled(True):
        model.train()
        batch_size = args.batch_size

        ninput, ntext, naudio, nlabel = next(nloader)
        ainput, atext, aaudio, alabel = next(aloader)

        input = torch.cat((ninput, ainput), 0).to(device)
        text = torch.cat((ntext, atext), 0).to(device)
        audio = torch.cat((naudio, aaudio), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input, text, audio)

        scores = scores.view(batch_size * 32 * 2, -1)
        scores = scores.squeeze()
        abn_scores = scores[batch_size * 32:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = RTFM_loss(args.alpha, 100, args.normal_weight, args.abnormal_weight)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        if args.extra_loss:
            cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal,
                                  feat_select_abn) + loss_smooth + loss_sparse
        else:
            cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)

        viz.plot_lines('loss', cost.item())
        viz.plot_lines('smooth loss', loss_smooth.item())
        viz.plot_lines('sparsity loss', loss_sparse.item())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
