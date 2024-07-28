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
import torch.utils.data
import torch.utils.data.distributed

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.utils import save_image

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from tools import *
from dataset import *


_MODELS = {
    'RN50': 'RN50',
    'RN101': 'RN101',
    'ViTB16': 'ViT-B/16',
    'ViTB32': 'ViT-B/32'
}


class CLIP:
    PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
    PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


# PIL reader
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        with Image.open(path) as img:
            return img.convert('RGB')


def main():
    set_random_seed(opt.manual_seed)

    # This codebase has only been tested under the single GPU setting
    assert opt.gpu is not None
    main_worker(opt.gpu, opt)


def main_worker(gpu, opt):
    opt.gpu = gpu
    set_random_seed(opt.manual_seed)


    test_seen_dataset = FeatureSet(opt, mode='seen')
    test_seen_features = test_seen_dataset.features
    test_seen_labels = test_seen_dataset.labels

    test_unseen_dataset = FeatureSet(opt, mode='unseen')
    test_unseen_features = test_unseen_dataset.features
    test_unseen_labels = test_unseen_dataset.labels

    print("evaluating: {}".format(opt.dataset))

    seenclasses = test_seen_dataset.seenclasses
    nseenclasses = test_seen_dataset.nseenclasses
    unseenclasses = test_seen_dataset.unseenclasses
    nunseenclasses = test_seen_dataset.nunseenclasses
    classnames = test_seen_dataset.classnames
    nclasses = test_seen_dataset.nclasses

    cls_features = torch.from_numpy(test_seen_dataset.cls_features).float()

    seen_logits = F.normalize(test_seen_features) @ F.normalize(cls_features).t()
    _, prediction = torch.max(seen_logits.data, 1)
    ground_truth = test_seen_labels
    classes = ground_truth.unique()
    acc_per_class = torch.FloatTensor(classes.size(0)).fill_(0)
    for i in range(classes.size(0)):
        idx = (ground_truth == classes[i])
        acc_per_class[i] = torch.sum(ground_truth[idx] == prediction[idx]) / torch.sum(idx)
    acc_seen = acc_per_class.mean()

    unseen_logits = F.normalize(test_unseen_features) @ F.normalize(cls_features).t()
    _, prediction = torch.max(unseen_logits.data, 1)
    ground_truth = test_unseen_labels
    classes = ground_truth.unique()
    acc_per_class = torch.FloatTensor(classes.size(0)).fill_(0)
    for i in range(classes.size(0)):
        idx = (ground_truth == classes[i])
        acc_per_class[i] = torch.sum(ground_truth[idx] == prediction[idx]) / torch.sum(idx)
    acc_unseen = acc_per_class.mean()

    acc_H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
    print(
        f'Zero-Shot CLIP per class accuracy: Seen = {acc_seen * 100}%, Unseen = {acc_unseen * 100}%, Harmony = {acc_H * 100}%')

    base_logits = F.normalize(test_seen_features) @ F.normalize(cls_features[:nseenclasses]).t()
    _, prediction = torch.max(base_logits.data, 1)
    ground_truth = test_seen_labels
    acc_base = torch.sum(prediction == ground_truth) / ground_truth.size(0)

    novel_logits = F.normalize(test_unseen_features) @ F.normalize(cls_features[nseenclasses:]).t()
    _, prediction = torch.max(novel_logits.data, 1)
    ground_truth = test_unseen_labels - nseenclasses
    acc_novel = torch.sum(prediction == ground_truth) / ground_truth.size(0)

    acc_HM = 2 * acc_base * acc_novel / (acc_base + acc_novel)
    print(f'Zero-Shot CLIP accuracy: Base = {acc_base * 100}%, Novel = {acc_novel * 100}%, Harmony = {acc_HM * 100}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prompt Tuning')
    parser.add_argument('--config', type=str, default='/data/ylong/dfgzsl2/pt/configs/AWA2.yaml', help='/path/to/config/file')
    args = parser.parse_args()

    config_filepath = args.config
    # load config file
    with open(config_filepath) as f:
        opt = yaml.load(f, yaml.CLoader)
        config_file = yaml.load(f, yaml.CLoader)

    # ultilize easydict to read config file
    opt = easydict.EasyDict(opt)

    main()
