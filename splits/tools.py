import os
import math
import json
from collections import OrderedDict
import errno
import pickle
import warnings
import random
import pdb


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def check_isfile(fpath):
    """Check if the given path is a file.
    Args:
        fpath (str): file path.
    Returns:
       bool
    """
    isfile = os.path.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile

def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.
    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items

def subsample_classes(*args, subsample="all"):
    """Divide classes into two groups. The first group
    represents base classes while the second group represents
    new classes.

    Args:
        args: a list of datasets, e.g. train, val and test.
        subsample (str): what classes to subsample.
    """
    assert subsample in ["all", "base", "new"]

    if subsample == "all":
        return args

    dataset = args[0]
    labels = set()
    for item in dataset:
        # labels.add(item.label)
        labels.add(item[1])
    labels = list(labels)
    # print("all:",labels)
    labels.sort()
    n = len(labels)
    # Divide classes into two halves
    m = math.ceil(n / 2)
    print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
    if subsample == "base":
        selected = labels[:m]  # take the first half
        print(selected)
    else:
        selected = labels[m:]  # take the second half
        print(selected)
    # relabeler = {y: y_new for y_new, y in enumerate(selected)}

    output = []
    for dataset in args:
        dataset_new = []
        for item in dataset:
            if item[1] not in selected:
                continue
            item_new = (item[0], item[1], item[2])
            dataset_new.append(item_new)
        output.append(dataset_new)

    return output


def read_split(filepath, path_prefix):
    def _convert(items):
        out = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            item = (impath, int(label), classname)
            out.append(item)
        return out

    print(f"Reading split from {filepath}")
    split = read_json(filepath)
    train = _convert(split["train"])
    val = _convert(split["val"])
    test = _convert(split["test"])

    return train, val, test


def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
    # The data are supposed to be organized into the following structure
    # =============
    # images/
    #     dog/
    #     cat/
    #     horse/
    # =============
    categories = listdir_nohidden(image_dir)
    categories = [c for c in categories if c not in ignored]
    categories.sort()

    p_tst = 1 - p_trn - p_val
    print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

    def _collate(ims, y, c):
        items = []
        for im in ims:
            item = (im, y, c)
            items.append(item)
        return items

    train, val, test = [], [], []
    for label, category in enumerate(categories):
        category_dir = os.path.join(image_dir, category)
        images = listdir_nohidden(category_dir)
        images = [os.path.join(category_dir, im) for im in images]
        random.shuffle(images)
        n_total = len(images)
        n_train = round(n_total * p_trn)
        n_val = round(n_total * p_val)
        n_test = n_total - n_train - n_val
        assert n_train > 0 and n_val > 0 and n_test > 0

        if new_cnames is not None and category in new_cnames:
            category = new_cnames[category]

        train.extend(_collate(images[:n_train], label, category))
        val.extend(_collate(images[n_train : n_train + n_val], label, category))
        test.extend(_collate(images[n_train + n_val :], label, category))

    return train, val, test


def save_split(train, val, test, filepath, path_prefix):
    def _extract(items):
        out = []
        for item in items:
            # print(type(item))
            # pdb.set_trace()
            impath = item[0]
            label = item[1]
            classname = item[2]
            impath = impath.replace(path_prefix, "")
            if impath.startswith("/"):
                impath = impath[1:]
            out.append((impath, label, classname))
        return out

    train = _extract(train)
    val = _extract(val)
    test = _extract(test)

    split = {"train": train, "val": val, "test": test}

    write_json(split, filepath)
    print(f"Saved split to {filepath}")