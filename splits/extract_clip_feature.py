import pdb

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import os
import argparse
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import clip
import re
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='name of datasets')
parser.add_argument('--dataroot', default='xxxx', help='root path of datasets')
parser.add_argument('--image_embedding', default='res101',help='from res101 to ViTB16')
parser.add_argument('--class_embedding', default='att',help='from att to clip')
parser.add_argument('--clip_embedding', default='RN101',help='from RN101 to ViT-B/16')

###

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "AWA2": "a photo of a {}, a type of animal.",
    "CUB": "a photo of a {}, a type of bird in North America.",
    "SUN": "a photo of a {}.",
    "aPY": "a photo of a {}.",
}

# Generate path
def get_path(image_files):
    image_files = np.squeeze(image_files)
    new_image_files = []
    for image_file in image_files:
        image_path = image_file[0]
        image_path = os.path.join(opt.dataroot, opt.dataset, 'images', '/'.join(image_path.split('/')[8:]))
        new_image_files.append(image_path)
    new_image_files = np.array(new_image_files)
    return new_image_files

def get_class(name):
    name = re.sub(pattern=r"\d|\.", repl=r"", string=name)
    name = re.sub(pattern=r"\+|_", repl=r" ", string=name)
    return name

# PIL reader
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:
        with Image.open(path) as img:
            return img.convert('RGB')

opt = parser.parse_args()

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = clip.load(opt.clip_embedding, device, jit=False)

clip_filename = opt.clip_embedding.replace('/', '') + '.mat'
clip_filename = clip_filename.replace('-', '')
print(clip_filename)

res_filename = opt.image_embedding + '.mat'
matcontent = sio.loadmat(os.path.join(opt.dataroot, opt.dataset, res_filename))

res_feature = matcontent['features'].T
image_files = get_path(matcontent['image_files'])

clip_feature = []
n_sample = res_feature.shape[0]
with torch.no_grad():
    for i in trange(n_sample, desc='image'):
        image = preprocess(pil_loader(image_files[i])).unsqueeze(0).to(device)
        img_embedding = model.encode_image(image)
        clip_feature.append(img_embedding.squeeze().cpu().numpy())

matcontent['features'] = np.array(clip_feature).T
clip_filename = opt.clip_embedding.replace('/', '') + '.mat'
clip_filename = clip_filename.replace('-', '')
print(clip_filename)
sio.savemat(os.path.join(opt.dataroot, opt.dataset, clip_filename), matcontent)


att_filename = opt.class_embedding + '_splits.mat'
matcontent = sio.loadmat(os.path.join(opt.dataroot, opt.dataset, att_filename))

attribute = matcontent['att'].T
class_names = []
items = matcontent['allclasses_names']
for item in items:
    name = item[0][0]
    if opt.dataset == 'AWA2':
        name = name.strip().replace('+', ' ')
    elif opt.dataset == 'CUB':
        name = name.strip().split('.')[1].replace('_', ' ')
    elif opt.dataset == 'SUN':
        name = name.strip().replace('_', ' ')
    class_names.append(name)
class_names = np.array(class_names)
print(class_names)

clip_cls_text = []
clip_cls_feature = []
n_class = attribute.shape[0]
template = CUSTOM_TEMPLATES[opt.dataset]
with torch.no_grad():
    for i in trange(n_class, desc='cls_text'):
        template.forat(class_names[i])
        clip_cls_text.append(template.forat(class_names[i]))
        text = clip.tokenize(template.forat(class_names[i])).to(device)
        text_embedding = model.encode_text(text)
        clip_cls_feature.append(text_embedding.squeeze().cpu().numpy())

matcontent['cls_text'] = np.array(clip_cls_text)
matcontent['cls_features'] = np.array(clip_cls_feature).T

clip_filename = 'clip_splits.mat'
sio.savemat(os.path.join(opt.dataroot, opt.dataset, clip_filename), matcontent)
