from typing import Optional
from torch.optim.optimizer import Optimizer

class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        counter = 0
        for param_group in self.optimizer.param_groups:
            if counter == 0:
                param_group['lr_mult'] = 0.1
            else:
                param_group['lr_mult'] = 1.
            # if 'lr_mult' not in param_group:
            #     param_group['lr_mult'] = 1.
            counter += 1
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


class WarmUpLR:

    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, counter=0):
        self.init_lr = init_lr
        self.gamma = gamma
        self.optimizer = optimizer
        self.iter_num = 0
        self.counter= counter

    def get_lr(self) -> float:
        lr = self.init_lr * (self.iter_num * self.gamma)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        counter = 0
        for param_group in self.optimizer.param_groups:
            if counter <= self.counter:
                param_group['lr_mult'] = 0.1
            else:
                param_group['lr_mult'] = 1.
            # if 'lr_mult' not in param_group:
            #     param_group['lr_mult'] = 1.
            counter += 1
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


class MyStepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, counter = 0):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.counter = counter

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        counter = 0
        for param_group in self.optimizer.param_groups:
            if counter <= self.counter:
                param_group['lr_mult'] = 0.1
            else:
                param_group['lr_mult'] = 1.
            # if 'lr_mult' not in param_group:
            #     param_group['lr_mult'] = 1.
            counter += 1
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


def adjust_learning_rate_inv(lr, optimizer, iters, alpha=0.001, beta=0.75):
    lr = lr / pow(1.0 + alpha * iters, beta)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10, power=0.75, init_lr=0.001, max_iter=10000):
    #10000
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    #max_iter = 10000
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return optimizer

def stepwiseDecaySheduler(step, initial_lr, gamma=0.001, power=0.75):
    return initial_lr * (1 + gamma * step) ** (- power)
