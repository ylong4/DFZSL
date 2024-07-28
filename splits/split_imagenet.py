import pdb
import torch
import scipy.io as sio
import numpy as np
import os
import pickle
import ipdb
from tools import *


def read_classnames(text_file):
    """Return a dictionary containing
    key-value pairs of <folder name>: <class name>.
    """
    classnames = OrderedDict()
    with open(text_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classname = " ".join(line[1:])
            classnames[folder] = classname
    return classnames


def read_data(image_dir, classnames, split_dir):
    split_dir = os.path.join(image_dir, split_dir)
    folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
    items = []

    for label, folder in enumerate(folders):
        imnames = listdir_nohidden(os.path.join(split_dir, folder))
        classname = classnames[folder]
        for imname in imnames:
            impath = os.path.join(split_dir, folder, imname)
            item = (impath, int(label), classname)
            items.append(item)

    return items


root = "/DFGZL/datasets/"
dataset_dir = os.path.join(root, "imagenet")
image_dir = os.path.join(dataset_dir, "image")
preprocessed = os.path.join(dataset_dir, "tang_preprocessed.pkl")
split_fewshot_dir = os.path.join(dataset_dir, "split_fewshot")
mkdir_if_missing(split_fewshot_dir)

if os.path.exists(preprocessed):
    with open(preprocessed, "rb") as f:
        preprocessed = pickle.load(f)
        train = preprocessed["train"]
        test = preprocessed["test"]
else:
    text_file = os.path.join(dataset_dir, "classnames.txt")
    classnames = read_classnames(text_file)
    train = read_data(image_dir, classnames, "train")
    # Follow standard practice to perform evaluation on the val set
    # Also used as the val set (so evaluate the last-step model)
    test = read_data(image_dir, classnames, "val")

    preprocessed_content = {"train": train, "test": test}
    with open(preprocessed, "wb") as f:
        pickle.dump(preprocessed_content, f, protocol=pickle.HIGHEST_PROTOCOL)

base_train, base_test = subsample_classes(train, test, subsample='base')
new_train, new_test = subsample_classes(train, test, subsample='new')
print("SUM",len(base_train)+len(base_test)+len(new_test))
image_files = np.hstack((np.array(base_train)[:, 0], np.array(base_test)[:, 0], np.array(new_test)[:, 0])).astype(str)
labels = np.hstack((np.array(base_train)[:, 1], np.array(base_test)[:, 1], np.array(new_test)[:, 1])).astype(int)
names = np.hstack((np.array(base_train)[:, 2], np.array(base_test)[:, 2], np.array(new_test)[:, 2])).astype(str)

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
print("len(allclasses_names)",len(allclasses_names))
ipdb.set_trace()


mat1 = {}
mat1['image_files'] = image_files
mat1['labels'] = (labels + 1).T

train_loc = np.arange(1, 1 + len(base_train))
val_loc = np.arange(1, 1 + len(base_train))
trainval_loc = np.arange(1, 1 + len(base_train))
test_seen_loc = np.arange(1 + len(base_train), 1 + len(base_train) + len(base_test))
test_unseen_loc = np.arange(1 + len(base_train) + len(base_test), 1 + len(base_train) + len(base_test) + len(new_test))
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
