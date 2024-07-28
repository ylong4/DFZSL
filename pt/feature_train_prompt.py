import argparse
import pdb
import sys,os
d = os.path.dirname(__file__)
parent_path = os.path.dirname(os.path.abspath(d))
sys.path.append(parent_path)

import yaml
import easydict

from copy import deepcopy

from PIL import Image
import numpy as np
import scipy.io as sio
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data
import torch.utils.data.distributed

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.utils import save_image

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.coop import get_coop
from clip.cocoop import get_cocoop
from clip.vlpt import get_vlpt
from tools import *
from dataset import *


_MODELS = {
    'RN50': 'RN50',
    'RN101': 'RN101',
    'ViTB16': 'ViT-B/16',
    'ViTB32': 'ViT-B/32'
}


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        min_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.min_lr = min_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


class CLIP:
    PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
    PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):
        super().__init__()
        # print('=========using KL Loss with temperature 10 and mean reduction=========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


# PIL reader
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        with Image.open(path) as img:
            return img.convert('RGB')


def save_mat(opt, model, arch):
    device = opt.gpu
    # _, _, preprocess = clip.load(arch, device)

    model.train(False)
    root = opt.dataroot
    dataset = opt.dataset
    matcontent = sio.loadmat(os.path.join(root, 'datasets', dataset, opt.image_embedding + '.mat'))
    # image_files = get_path(matcontent['image_files'])

    all_dataset = FeatureSet(opt, mode='all')
    print("number of train samples: {}".format(len(all_dataset)))
    all_loader = torch.utils.data.DataLoader(
        all_dataset, drop_last=False,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)
    seenclasses = all_dataset.seenclasses
    unseenclasses = all_dataset.unseenclasses
    allclasses = all_dataset.allclasses

    iters = tqdm(all_loader, desc=f'saving ', total=len(all_loader))
    new_features = []
    with torch.no_grad():
        for i, (img, _) in enumerate(iters):
            img = img.cuda(opt.gpu, non_blocking=True)
            image_features = model.get_image_features(img, image=False)
            new_features.append(image_features.squeeze().cpu())
        new_features = torch.cat(new_features, dim=0).numpy()

        text_features = model.get_text_features()
        text_features = text_features.squeeze().cpu().numpy()

        new_cls_features = []
        for i in range(len(allclasses)):
            idx = (allclasses == i)
            new_cls_features.append(np.squeeze(text_features[idx]))
        new_cls_features = np.array(new_cls_features)

    matcontent['features'] = new_features.T
    sio.savemat(os.path.join(root, 'datasets', dataset, opt.experiment + '-' + opt.image_embedding + '.mat'), matcontent)

    matcontent = sio.loadmat(os.path.join(root, 'datasets', opt.dataset, opt.class_embedding + '_splits.mat'))
    matcontent['cls_features'] = new_cls_features.T
    sio.savemat(os.path.join(root, 'datasets', dataset, opt.experiment + '-' + opt.class_embedding + '_splits.mat'), matcontent)


def main():
    set_random_seed(opt.manual_seed)

    # This codebase has only been tested under the single GPU setting
    assert opt.gpu is not None
    main_worker(opt.gpu, opt)


def main_worker(gpu, opt):
    opt.gpu = gpu
    set_random_seed(opt.manual_seed)

    train_dataset = FeatureSet(opt, mode='train')
    print("number of train samples: {}".format(len(train_dataset)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, drop_last=False,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)

    test_seen_dataset = FeatureSet(opt, mode='seen')
    print("number of test seen samples: {}".format(len(test_seen_dataset)))
    test_seen_loader = torch.utils.data.DataLoader(
        test_seen_dataset, drop_last=False,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)

    test_unseen_dataset = FeatureSet(opt, mode='unseen')
    print("number of test unseen samples: {}".format(len(test_unseen_dataset)))
    test_unseen_loader = torch.utils.data.DataLoader(
        test_unseen_dataset, drop_last=False,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)

    print("evaluating: {}".format(opt.dataset))

    seenclasses = train_dataset.seenclasses
    nseenclasses = train_dataset.nseenclasses
    unseenclasses = train_dataset.unseenclasses
    nunseenclasses = train_dataset.nunseenclasses
    classnames = train_dataset.classnames
    nclasses = train_dataset.nclasses

    cls_features = torch.from_numpy(train_dataset.cls_features).float()

    arch = _MODELS[opt.image_embedding.split('-')[-1].strip()]

    print("=> Model created: visual backbone {}".format(arch))
    if opt.experiment == 'vlpt':
        model = get_vlpt(opt.gpu, classnames, opt.residual_weight, arch, opt.n_ctx, opt.ctx_init)
        to_update = ["prompt_learner.ctx", "prompt_learner.t2i"]
    elif opt.experiment == 'coop':
        model = get_coop(opt.gpu, classnames, arch, opt.n_ctx, opt.ctx_init)
        to_update = ["prompt_learner.ctx"]
    else:
        model = get_cocoop(opt.gpu, classnames, arch, opt.n_ctx, opt.ctx_init)
        to_update = ["prompt_learner.ctx", "prompt_learner.meta_net"]

    print("Use GPU: {} for training".format(opt.gpu))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert opt.gpu is not None
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)

    for name, param in model.named_parameters():
        flag = True
        for part_name in to_update:
            flag = flag and (part_name not in name)
        if flag:
            param.requires_grad_(False)

    # define optimizer
    params_to_update = []
    to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            to_update.append(name)
    print(to_update)
    optimizer = torch.optim.AdamW(params_to_update, opt.lr)
    # optimizer = torch.optim.SGD(params_to_update, lr=opt.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
    # scheduler = ConstantWarmupScheduler(optimizer, scheduler, 1, 1e-5)
    image_criterion = KLLoss()
    text_criterion = KLLoss()

    # iterating through eval datasets
    per_class_results = []
    results = []

    for epoch in range(opt.nepoch):
        print('=' * 50)

        model.train(True)
        iters = tqdm(train_loader, desc=f'epoch {epoch}/{opt.nepoch} ', total=len(train_loader))
        for i, (img, target) in enumerate(iters):
            assert opt.gpu is not None
            with torch.cuda.amp.autocast():
                img = img.cuda(opt.gpu, non_blocking=True)
                target = target.cuda(opt.gpu, non_blocking=True)
                np_target = target.cpu().numpy()
                gt = np.zeros(shape=(len(np_target), nclasses))
                for i in range(len(np_target)):
                    gt[i][np_target[i]] = 1
                sampled_classes = np.array([i for i in range(nseenclasses)])
                # sampled_classes, inverse_ind = np.unique(np_target, return_inverse=True)
                gt = gt[:, sampled_classes]
                ground_truth = torch.from_numpy(gt).float().to(target.device)

                if opt.experiment == 'cocoop':
                    logits = model.get_logits(img, image=False)
                    logits = logits[:, sampled_classes]
                    loss = F.cross_entropy(logits, ground_truth)
                else:
                    image_features, text_features = model.get_features(img, image=False)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    logit_scale = model.logit_scale.exp()

                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    # logits_per_text = logit_scale * text_features @ image_features.t()
                    # loss_image = image_criterion(logits_per_image[:, sampled_classes], ground_truth)
                    # loss_text = text_criterion(logits_per_text[sampled_classes, :], ground_truth.t())
                    # loss = (loss_image + loss_text) / 2

                    logits = logit_scale * image_features @ text_features.t()
                    logits = logits[:, sampled_classes]
                    loss = F.cross_entropy(logits, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        print(f'total loss: {loss.item():6.2f}')

        S, B = evaluate(test_seen_loader, model, 'seen', nseenclasses, opt)
        U, N = evaluate(test_unseen_loader, model, 'unseen', nseenclasses, opt)
        H = (2 * S * U) / (S + U)
        HM = (2 * B * N) / (B + N)
        per_class_results.append((S, U, H))
        results.append((B, N, HM))

    best_per_class_results = sorted(per_class_results, key=lambda x: x[2], reverse=True)[0]
    best_results = sorted(results, key=lambda x:x[2], reverse=True)[0]
    del train_dataset, train_loader, test_seen_dataset, test_seen_loader, test_unseen_dataset, test_unseen_loader

    # save_mat(opt, model, arch)
    # np.save(os.path.join(opt.dataroot, 'datasets', opt.dataset, opt.experiment + '-' + opt.image_embedding + '-results'), np.array(results))

    print("======== Per-class Result Summary ========")
    print("  [dataset] \t accS@1  accU@1  accH@1")
    print(
        f"{opt.dataset:>10} \t {100 * best_per_class_results[0]:5.2f}%  {100 * best_per_class_results[1]:5.2f}%  {100 * best_per_class_results[2]:5.2f}%")
    print(f"experiment {opt.experiment}  {opt.image_embedding}")
    print("======== Base2New Result Summary ========")
    print("  [dataset] \t accB@1  accN@1  accHM@1")
    print(
        f"{opt.dataset:>10} \t {100 * best_results[0]:5.2f}%  {100 * best_results[1]:5.2f}%  {100 * best_results[2]:5.2f}%")


def evaluate(test_loader, model, split, nclasses, opt):
    model.train(False)
    iters = tqdm(test_loader, desc=f'testing ', total=len(test_loader))
    predicted, base2new_predicted, label = [], [], []
    with torch.no_grad():
        for i, (img, target) in enumerate(iters):
            assert opt.gpu is not None
            img = img.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

            if opt.experiment == 'cocoop':
                logits = model.get_logits(img, image=False)
            else:
                image_features, text_features = model.get_features(img, image=False)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

            predicted.extend(list(torch.max(logits.data, 1)[1].cpu().numpy()))
            if split == 'seen':
                base2new_predicted.extend(list(torch.max(logits.data[:, :nclasses], 1)[1].cpu().numpy()))
            else:
                base2new_predicted.extend(list(torch.max(logits.data[:, nclasses:], 1)[1].cpu().numpy()))
            label.extend(list(target.cpu().numpy()))

    label = np.array(label)

    predicted = np.array(predicted)
    target_classes = np.unique(label)
    acc_per_class = 0
    for i in target_classes:
        idx = (label == i)
        acc_per_class += np.sum(label[idx] == predicted[idx]) / np.sum(idx)
    acc_per_class /= target_classes.shape[0]
    print(f'test {split} per class accuracy: {100 * acc_per_class:.2f}%')

    base2new_predicted = np.array(base2new_predicted)
    if split == 'seen':
        acc = np.sum(base2new_predicted == label) / len(label)
    else:
        acc = np.sum(base2new_predicted == (label - nclasses)) / len(label)
    print(f'test {split} base2new accuracy: {100 * acc:.2f}%')

    return acc_per_class, acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prompt Tuning')
    parser.add_argument('--config', type=str, default='/DFGZL/pt/configs/GZSL/AWA2.yaml', help='/path/to/config/file')
    args = parser.parse_args()

    config_filepath = args.config
    with open(config_filepath) as f:
        opt = yaml.load(f, yaml.CLoader)
        config_file = yaml.load(f, yaml.CLoader)

    opt = easydict.EasyDict(opt)
    if opt.experiment == "cocoop":
        opt.batch_size = 1
    main()
