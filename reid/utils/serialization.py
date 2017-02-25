from __future__ import print_function
import json
import os.path as osp
import shutil

import torch


def read_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_model(state, is_best, fpath='checkpoint.pth.tar'):
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_model_(fpath, model):
    if osp.isfile(fpath):
        print("=> Loading checkpoint '{}'".format(fpath))
        checkpoint = torch.load(fpath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
