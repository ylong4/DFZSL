import pdb
import sys,os
import torch

d = os.path.dirname(__file__)
parent_path = os.path.dirname(os.path.abspath(d))
sys.path.append(parent_path)
import torch.nn as nn
import torch.nn.functional as F
import utils

import clip

import numpy as np


class Classifier(nn.Module):
    def __init__(self, input_dim, nclass, metric, proto=None, activation='None'):
        super(Classifier, self).__init__()
        scale = input_dim ** -0.5
        if proto is None:
            self.weights = nn.Parameter(scale * torch.randn(nclass, input_dim))
        else:
            self.weights = nn.Parameter(proto)
        self.activation = utils.activation_dict[activation]
        self.metric = metric

    def forward(self, x, tem=1.0):
        if self.activation is not None:
            w = self.activation(self.weights)
        else:
            w = self.weights
        if self.metric == 'euclidean':
            f = x
        elif self.metric == 'cosine':
            f = F.normalize(x) # x is in shape [batch_size, n_dimension], default dim for F.normalize is 1
            w = F.normalize(w)
        x = f @ w.t() / tem
        return x


class Simulator(nn.Module):
    def __init__(self, nclass, ndim, proto, concentration, learned_concentration=False):
        super(Simulator, self).__init__()
        scale = ndim ** -0.5
        self.learned_concentration = learned_concentration
        if proto is not None:
            self.mu = nn.Parameter(proto)
        else:
            self.mu = nn.Parameter(scale * torch.randn(nclass, ndim))

        if learned_concentration:
            self.concentration = nn.Parameter(concentration)
        else:
            self.concentration = nn.Parameter(concentration, requires_grad=False)


    def forward(self):
        mu = F.normalize(self.mu)

        if self.learned_concentration:
            concentration = self.concentration
        else:
            concentration = F.softplus(self.concentration)

        return mu, concentration


class Generator(nn.Module):
    def __init__(self, condition_dim, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.condition_dim = condition_dim
        self.noise_dim = noise_dim
        self.fc1 = nn.Linear(condition_dim+noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(utils.weights_init)

    def forward(self, condition, noise):
        input = torch.cat((condition, noise), dim=1)
        hidden = self.lrelu(self.fc1(input))
        generation = self.fc2(hidden)
        return generation
