import os
import random
import scipy.io as sio
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.utils.data as data
from preprocess import *


class ImageSet(data.Dataset):
    def __init__(self, opt, mode='train', transform=None, n_shot=None):
        self.root = opt.dataset.root
        self.dataset = opt.dataset.name
        self.image_embedding = opt.dataset.image_embedding
        self.class_embedding = opt.dataset.class_embedding
        self.mode = mode
        self.transform = transform
        self.n_shot = None

        matcontent = sio.loadmat(os.path.join(self.root, 'datasets', self.dataset, self.image_embedding + '.mat'))
        image_files = self.get_path(matcontent['image_files'])
        labels = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(os.path.join(self.root, 'datasets', self.dataset, self.class_embedding + '_splits.mat'))
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        train_seen_label = labels[trainval_loc].astype(int)
        # train_seen_label = labels[train_loc].astype(int)
        test_seen_label = labels[test_seen_loc].astype(int)
        self.seenclasses = np.unique(test_seen_label)
        self.nseenclasses = len(self.seenclasses)
        test_unseen_label = labels[test_unseen_loc].astype(int)
        self.unseenclasses = np.unique(test_unseen_label)
        self.nunseenclasses = len(self.unseenclasses)
        self.nclasses = self.nseenclasses + self.nunseenclasses
        self.allclasses = np.hstack((self.seenclasses, self.unseenclasses))

        cls_features = matcontent['cls_features'].T
        self.cls_features = cls_features[self.allclasses]

        train_seen_label = map_label(torch.from_numpy(train_seen_label).long(), torch.from_numpy(self.seenclasses).long()).numpy()
        test_seen_label = map_label(torch.from_numpy(test_seen_label).long(), torch.from_numpy(self.seenclasses).long()).numpy()
        test_unseen_label = map_label(torch.from_numpy(test_unseen_label).long(), torch.from_numpy(self.unseenclasses).long()).numpy() + self.nseenclasses

        cname = []
        allclasses_names = matcontent['allclasses_names']
        for item in allclasses_names:
            name = item[0][0]
            if self.dataset == 'AWA2':
                name = name.strip().replace('+', ' ')
            elif self.dataset == 'CUB':
                name = name.strip().split('.')[1].replace('_', ' ')
            elif self.dataset == 'SUN':
                name = name.strip().replace('_', ' ')
            cname.append(name)
        self.classnames = np.array(cname)[self.allclasses]

        if self.mode == 'train':
            self.image_list = list(image_files[trainval_loc])
            # self.image_list = list(image_files[train_loc])
            self.label_list = list(train_seen_label)
        elif self.mode == 'seen':
            self.image_list = list(image_files[test_seen_loc])
            self.label_list = list(test_seen_label)
        elif self.mode == 'unseen':
            self.image_list = list(image_files[test_unseen_loc])
            self.label_list = list(test_unseen_label)
        else:
            self.image_list = list(image_files)
            self.label_list = list(labels)

        if n_shot is not None:
            few_shot_samples = []
            c_range = np.unique(self.labels.numpy())
            for c in c_range:
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]


    def get_path(self, image_files):
        image_files = np.squeeze(image_files)
        new_image_files = []
        for image_file in image_files:
            image_file = image_file[0]
            image_file = '/'.join(image_file.split('/')[8:])
            new_image_files.append(image_file)
        new_image_files = np.array(new_image_files)
        return new_image_files

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, 'datasets', self.dataset, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()


class SimpleFeatureSet(data.Dataset):
    def __init__(self,  features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        feat = self.features[index]
        label = self.labels[index]
        return feat, label

    def __len__(self):
        return len(self.labels)


class FeatureSet(data.Dataset):
    def __init__(self, opt, mode='train', n_shot=None):
        self.root = opt.dataset.root
        self.dataset = opt.dataset.name
        self.image_embedding = opt.dataset.image_embedding
        self.class_embedding = opt.dataset.class_embedding
        self.mode = mode
        self.n_shot = None

        matcontent = sio.loadmat(os.path.join(self.root, 'datasets', self.dataset, self.image_embedding + '.mat'))
        features = matcontent['features'].T
        labels = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(os.path.join(self.root, 'datasets', self.dataset, self.class_embedding + '_splits.mat'))
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        train_seen_feature = features[trainval_loc]
        train_seen_label = labels[trainval_loc].astype(int)
        # train_seen_feature = features[train_loc]
        # train_seen_label = labels[train_loc].astype(int)
        test_seen_feature = features[test_seen_loc]
        test_seen_label = labels[test_seen_loc].astype(int)
        test_unseen_feature = features[test_unseen_loc]
        test_unseen_label = labels[test_unseen_loc].astype(int)

        self.seenclasses = np.unique(test_seen_label)
        self.nseenclasses = len(self.seenclasses)
        self.unseenclasses = np.unique(test_unseen_label)
        self.nunseenclasses = len(self.unseenclasses)
        self.nclasses = self.nseenclasses + self.nunseenclasses
        self.allclasses = np.hstack((self.seenclasses, self.unseenclasses))

        cls_features = matcontent['cls_features'].T
        # cls_features = matcontent['att'].T
        self.cls_features = cls_features[self.allclasses]

        train_seen_label = map_label(torch.from_numpy(train_seen_label).long(), torch.from_numpy(self.seenclasses).long()).numpy()
        test_seen_label = map_label(torch.from_numpy(test_seen_label).long(), torch.from_numpy(self.seenclasses).long()).numpy()
        test_unseen_label = map_label(torch.from_numpy(test_unseen_label).long(), torch.from_numpy(self.unseenclasses).long()).numpy() + self.nseenclasses

        cname = []
        allclasses_names = matcontent['allclasses_names']
        for item in allclasses_names:
            name = item[0][0]
            if self.dataset == 'AWA2':
                name = name.strip().replace('+', ' ')
            elif self.dataset == 'CUB':
                name = name.strip().split('.')[1].replace('_', ' ')
            elif self.dataset == 'SUN':
                name = name.strip().replace('_', ' ')
            cname.append(name)
        self.classnames = np.array(cname)[self.allclasses]

        if self.mode == 'train':
            self.features = torch.from_numpy(train_seen_feature).float()
            self.labels = torch.from_numpy(train_seen_label).long()
        elif self.mode == 'seen':
            self.features = torch.from_numpy(test_seen_feature).float()
            self.labels = torch.from_numpy(test_seen_label).long()
        elif self.mode == 'unseen':
            self.features = torch.from_numpy(test_unseen_feature).float()
            self.labels = torch.from_numpy(test_unseen_label).long()
        else:
            self.features = torch.from_numpy(features).float()
            self.labels = torch.from_numpy(labels).long()

        if n_shot is not None:
            few_shot_samples = []
            c_range = np.unique(self.labels.numpy())
            for c in c_range:
                c_idx = [idx for idx, lable in enumerate(list(self.labels.numpy())) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.features = torch.stack([self.features[i] for i in few_shot_samples], dim=0)
            self.labels = torch.LongTensor([self.labels[i] for i in few_shot_samples])


    def __len__(self):
        return self.labels.size(0)


    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label