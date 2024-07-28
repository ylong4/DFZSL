import pdb
import torch
import scipy.io as sio
from scipy.io import loadmat
import numpy as np
import os
from collections import OrderedDict
import errno
import pickle
import json
import math
import random
import warnings
from collections import defaultdict
from tools import *


def read_data(label_file, lab2cname_file, image_dir):
    tracker = defaultdict(list)
    label_file = loadmat(label_file)["labels"][0]
    for i, label in enumerate(label_file):
        imname = f"image_{str(i + 1).zfill(5)}.jpg"
        impath = os.path.join(image_dir, imname)
        label = int(label)
        tracker[label].append(impath)

    print("Splitting data into 50% train, 20% val, and 30% test")

    def _collate(ims, y, c):
        items = []
        for im in ims:
            item = (im, y - 1, c)  # convert to 0-based label
            items.append(item)
        return items

    lab2cname = read_json(lab2cname_file)
    train, val, test = [], [], []
    for label, impaths in tracker.items():
        random.shuffle(impaths)
        n_total = len(impaths)
        print(len(impaths))
        pdb.set_trace()
        n_train = round(n_total * 0.5)
        n_val = round(n_total * 0.2)
        n_test = n_total - n_train - n_val
        assert n_train > 0 and n_val > 0 and n_test > 0
        cname = lab2cname[str(label)]
        train.extend(_collate(impaths[:n_train], label, cname))
        val.extend(_collate(impaths[n_train: n_train + n_val], label, cname))
        test.extend(_collate(impaths[n_train + n_val:], label, cname))

    return train, val, test


root = "/DFGZL/datasets/"
dataset_dir = os.path.join(root, "oxford_flowers")
image_dir = os.path.join(dataset_dir, "jpg")
label_file = os.path.join(dataset_dir, "imagelabels.mat")
lab2cname_file = os.path.join(dataset_dir, "cat_to_name.json")
split_path = os.path.join(dataset_dir, "split_zhou_OxfordFlowers.json")
split_fewshot_dir = os.path.join(dataset_dir, "split_fewshot")
mkdir_if_missing(split_fewshot_dir)

if os.path.exists(split_path):
    train, val, test = read_split(split_path, image_dir)
else:
    train, val, test = read_data(label_file, lab2cname_file, image_dir)
    save_split(train, val, test, split_path, image_dir)




base_train, base_val, base_test = subsample_classes(train, val, test, subsample='base')
new_train, new_val, new_test = subsample_classes(train, val, test, subsample='new')

print("train",len(train),"val",len(val),"test:",len(test))
print("base_train:",len(base_train),"new_train:",len(new_train))
print("base_val:",len(base_val),"new_val:",len(new_val))
print("base_test:",len(base_test),"new_test:",len(new_test))
print("SUM",len(base_train)+len(base_val)+len(base_test)+len(new_test))
pdb.set_trace()
# coop flower sum=4093=len(all_train)是巧合，不是失误



image_files = np.hstack((np.array(base_train)[:, 0], np.array(base_val)[:, 0],
                         np.array(base_test)[:, 0], np.array(new_test)[:, 0])).astype(str)
labels = np.hstack((np.array(base_train)[:, 1], np.array(base_val)[:, 1],
                    np.array(base_test)[:, 1], np.array(new_test)[:, 1])).astype(int)

names = np.hstack((np.array(base_train)[:, 2], np.array(base_val)[:, 2],
                   np.array(base_test)[:, 2], np.array(new_test)[:, 2])).astype(str)

labels_base_train= np.array(base_test)[:, 1].astype(int)
labels_test_new= np.array(new_test)[:, 1].astype(int)
labels_test_base= np.array(base_test)[:, 1].astype(int)
labels_test_all=np.hstack((np.array(base_test)[:, 1], np.array(new_test)[:, 1])).astype(int)

allclasses_names = []
unique_labels = list(np.unique(labels))
labels_base_train = list(np.unique(labels_base_train))
labels_test_new = list(np.unique(labels_test_new))
labels_test_base = list(np.unique(labels_test_base))
#997,是由于labels_test_new =497

pdb.set_trace()
unique_labels.sort()
for l in unique_labels:
    idx = (labels == l)
    allclasses_names.append([[names[idx][0]]])
allclasses_names = np.array(allclasses_names)
print("len(allclasses_names)",allclasses_names)
pdb.set_trace()


mat1 = {}
mat1['image_files'] = image_files
mat1['labels'] = (labels + 1).T

train_loc = np.arange(1, 1 + len(base_train))
val_loc = np.arange(1 + len(base_train), 1 + len(base_train) + len(base_val))
trainval_loc = np.arange(1, 1 + len(base_train) + len(base_val))
test_seen_loc = np.arange(1 + len(base_train) + len(base_val), 1 + len(base_train) + len(base_val) + len(base_test))
test_unseen_loc = np.arange(1 + len(base_train) + len(base_val) + len(base_test), 1 + len(base_train) + len(base_val) + len(base_test) + len(new_test))
save_path1=os.path.join(root, dataset_dir, 'ViTB16.mat')
sio.savemat(save_path1, mat1)

vdm_num=len(image_files[trainval_loc])/len(list(np.unique(labels[trainval_loc])))
print(dataset_dir,"vdm_num:",vdm_num)

mat2 = {}
mat2['allclasses_names'] = allclasses_names
mat2['trainval_loc'] = trainval_loc
mat2['train_loc'] = train_loc
mat2['val_loc'] = val_loc
mat2['test_seen_loc'] = test_seen_loc
mat2['test_unseen_loc'] = test_unseen_loc
save_path2=os.path.join(root, dataset_dir, 'clip_splits.mat')
sio.savemat(save_path2, mat2)
# 还需要提取图片feature cls_txt  文本feature