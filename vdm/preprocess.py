import torch
import numpy as np
from sklearn import preprocessing

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

def seed_everything(cuda, seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda != -1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def L2_norm(x, dim):
    '''
    Implement L2 normalization
    :param x: FloatTensor
    :param dim: 0 for Batch-norm & 1 for Layer-Norm
    :param keepdim: keep input size
    :return:
    '''
    n = torch.norm(x, dim=dim, keepdim=True).expand_as(x)
    o = x.div(n)
    return o

def Minmax_norm(x, dim):
    '''
    implement Minmax normalization
    :param x: FloatTensor
    :param dim: 0 for Batch-norm & 1 for Layer-Norm
    :param keepdim: keep input size
    :return:
    '''
    scaler = preprocessing.MinMaxScaler()
    x = x.numpy()
    if dim == 0:
        x = scaler.fit_transform(x)

    if dim == 1:
        x = x.T
        x = scaler.fit_transform(x)
        x = x.T
    return torch.from_numpy(x).float()

def Standard_norm(x, dim):
    '''
    implement Standard normalization
    :param x: FloatTensor
    :param dim: 0 for Batch-norm & 1 for Layer-Norm
    :param keepdim: keep input size
    :return:
    '''
    scaler = preprocessing.StandardScaler()
    x = x.numpy()
    if dim == 0:
        x = scaler.fit_transform(x)
    if dim == 1:
        x = x.T
        x = scaler.fit_transform(x)
        x = x.T
    return torch.from_numpy(x).float()
