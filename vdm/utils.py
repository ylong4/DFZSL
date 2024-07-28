import sys
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.distributions import MultivariateNormal
import os
import shutil
sys.path.append('.')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, logger, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class TestProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="124"):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)

    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        script_path = os.path.join(path, 'scripts')
        if os.path.exists(script_path):
            shutil.rmtree(script_path)
        os.mkdir(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            print(dst_file)
            shutil.copytree(script, dst_file)


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    err_str = "Batch dim of predictions and labels must match"
    assert preds.size(0) == labels.size(0), err_str
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].view(-1).float().sum() for k in ks]
    return [(1.0 - x / preds.size(0)) * 100.0 for x in topks_correct]

def get_optimizer(params, conf):
    if conf[0] == 'SGD':
        optimizer = optim.SGD(params,
                              lr=conf[1], momentum=conf[2],
                              weight_decay=conf[3], dampening=conf[4],
                              nesterov=conf[5])
    elif conf[0] == 'Adam':
        optimizer = torch.optim.Adam(params,
                                     lr=conf[1], betas=conf[2],
                                     weight_decay=conf[3], eps=conf[4],
                                     amsgrad=conf[5])
    elif conf[0] == 'AdamW':
        optimizer = torch.optim.AdamW(params,
                                      lr=conf[1], betas=conf[2],
                                      weight_decay=conf[3],eps=conf[4],
                                      amsgrad=conf[5])
    return optimizer

def get_multivariate_normal_samplers(mu, cov):
    samplers = []
    for i in range(mu.size(0)):
        samplers.append(MultivariateNormal(loc=mu[i], covariance_matrix=cov[i]))
    return samplers


def compute_covariance(x):
    '''
    get the co-variance of a sample matrix X
    :param x: torch.FloatTensor in size(nsample, ndim)
    :return:
    '''
    n = x.size(0)
    d = x.size(1)
    xb = x.mean(dim=0, keepdim=True).expand(n, d)

    return torch.mm((x-xb).t(), (x-xb)) / (n-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)

activation_dict = {'relu': nn.ReLU(),
                   'lrelu': nn.LeakyReLU(),
                   'sigmoid': nn.Sigmoid(),
                   'None': None}

if __name__ == "__main__":
    a = AverageMeter('a',':5.2f')
    p = TestProgressMeter(32, [a])
    a.update(3)
    a.update(11)
    a.update(9)
    p.display(20)
    print(a.avg)