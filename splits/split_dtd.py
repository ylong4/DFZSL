import pdb
import torch
import scipy.io as sio
import numpy as np
import os
from collections import OrderedDict
import errno
import pickle
import ipdb
import json
import math
import random
import warnings
from tools import *


root = "/DFGZL/datasets/"
dataset_dir = os.path.join(root, "dtd")
image_dir = os.path.join(dataset_dir, "images")
split_path = os.path.join(dataset_dir, "split_DescribableTextures.json")
split_fewshot_dir = os.path.join(dataset_dir, "split_fewshot")
mkdir_if_missing(split_fewshot_dir)

if os.path.exists(split_path):
    train, val, test = read_split(split_path, image_dir)
else:
    train, val, test = read_and_split_data(image_dir)
    save_split(train, val, test, split_path, image_dir)


base_train, base_val, base_test = subsample_classes(train, val, test, subsample='base')
new_train, new_val, new_test = subsample_classes(train, val, test, subsample='new')
print("SUM",len(base_train)+len(base_val)+len(base_test)+len(new_test))
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

ipdb.set_trace()
unique_labels.sort()
for l in unique_labels:
    idx = (labels == l)
    allclasses_names.append([[names[idx][0]]])
allclasses_names = np.array(allclasses_names)
print("len(allclasses_names)",allclasses_names)
ipdb.set_trace()


mat1 = {}
mat1['image_files'] = image_files
mat1['labels'] = (labels + 1).T

train_loc = np.arange(1, 1 + len(base_train))
val_loc = np.arange(1 + len(base_train), 1 + len(base_train) + len(base_val))
trainval_loc = np.arange(1, 1 + len(base_train) + len(base_val))
test_seen_loc = np.arange(1 + len(base_train) + len(base_val), 1 + len(base_train) + len(base_val) + len(base_test))
test_unseen_loc = np.arange(1 + len(base_train) + len(base_val) + len(base_test), 1 + len(base_train) + len(base_val) + len(base_test) + len(new_test))
save_path1=os.path.join(root, dataset_dir, 'ViTB16.mat')
# sio.savemat(save_path1, mat1)

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
# sio.savemat(save_path2, mat2)
