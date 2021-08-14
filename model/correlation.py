import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_CORR_torch(pred, real): #[sample, cate]
    x = pred[real.sum(-1) != 0]
    y = real[real.sum(-1) != 0]

    vx = x - torch.mean(x, dim=1, keepdim=True)
    vy = y - torch.mean(y, dim=1, keepdim=True)

    cost = torch.mean(torch.sum(vx * vy, dim=1, keepdim=True) / (torch.sqrt(torch.sum(vx ** 2, dim=1, keepdim=True)) * torch.sqrt(torch.sum(vy ** 2, dim=1, keepdim=True))))
    return cost

def get_CORR_torch2(pred, real): #[sample, cate]
    x = pred
    y = real

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.mean(vx * vy) / (torch.sqrt(torch.mean(vx ** 2)) * torch.sqrt(torch.mean(vy ** 2)))
    return cost



def cross_entropy(pred, real): #[sample, cate]
    x = pred[real.sum(-1) != 0]
    y = real[real.sum(-1) != 0]

    cost = - torch.mean(torch.sum(y * torch.log(x), dim=1))
    return cost


def get_CORR_numpy(pred, real): #[sample, cate]
    x = pred[real.sum(-1) != 0]
    y = real[real.sum(-1) != 0]

    vx = x - np.mean(x, axis=1)[:, np.newaxis]
    vy = y - np.mean(y, axis=1)[:, np.newaxis]

    cost = np.mean(np.sum(vx * vy, axis=1)[:, np.newaxis] / (np.sqrt(np.sum(vx ** 2, axis=1)[:, np.newaxis]) * np.sqrt(np.sum(vy ** 2, axis=1)[:, np.newaxis])))
    return cost


def get_CORR_numpy2(pred, real): #[sample, cate]
    x = pred
    y = real

    vx = x - np.mean(x)
    vy = y - np.mean(y)

    cost = np.mean(vx * vy) / (np.sqrt(np.mean(vx ** 2)) * np.sqrt(np.mean(vy ** 2)))
    return cost