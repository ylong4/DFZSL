import yaml
import os
import argparse
import random
import easydict
from preprocess import *


parser = argparse.ArgumentParser(description='Code for *Virtual Domain Modeling*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='DFGZL/vdm/configs/AWA2.yaml', help='/path/to/config/file')
args = parser.parse_args()

config_filepath = args.config
# load config file
with open(config_filepath) as f:
    opt = yaml.load(f, yaml.CLoader)
    config_file = yaml.load(f, yaml.CLoader)

# ultilize easydict to read config file
opt = easydict.EasyDict(opt)
opt.manual_seed=0

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
seed_everything(opt.cuda, opt.manual_seed)

if opt.cuda != -1:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("WARNING: You do not have a CUDA device, so you should probably run without --cuda.")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda)
        device = torch.device(f"cuda")
else:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")