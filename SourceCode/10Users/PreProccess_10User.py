#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import h5py
import json
from collections import defaultdict
from PIL import Image

def getLabel(filename):
#     input_string = "airpixels-123"
    dash_index = filename.find("-", 0)
    return filename[0:dash_index]
    

common_img_extensions = {'.tif', '.tiff', '.gif', '.jpeg', '.jpg', '.jif', '.jfif',
                         '.jp2', '.jpx', '.j2k', '.j2c', '.fpx', '.pcd', '.png'}


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


STANDARD_SIZE=(350,350)

def load_image(filename, verbose=False):
    img = Image.open(filename)
    if verbose==True:
        print( "(%s) changing size from %s to %s" % (filename, str(img.size), str(STANDARD_SIZE)) )
    img = img.resize(STANDARD_SIZE)
    res = np.array(img.getdata(), dtype=np.int16)
    if img.mode != 'RGB':
        if img.mode == 'RGBA':
            res = res[:,0:3]
        else:
            imgc = Image.new('RGB', img.size)
            imgc.paste(img)
            res = np.array(imgc.getdata(), dtype=np.int16)

    return res.reshape(STANDARD_SIZE+(3,))


def load(rootdir, include, val_percent = 0.15):
    print('loading facets and filenames from {}'.format(rootdir))
    imgs = defaultdict(set)
    
    for root,_,files in os.walk(rootdir):
        for f in files:
            extension = os.path.splitext(f)[1]
            if not extension in common_img_extensions:
                continue
            facet = getLabel(f)
            if facet in include:
                ffull = os.path.join(root,f)
                imgs[facet].add(ffull)
    X_res_train, y_res_train = [], []
    X_res_val, y_res_val = [], []
    
    for facet, files in imgs.items():
        X = []
        y = []
#         count = 0
        print( 'processing {}: {} files'.format(facet, len(files)) )
        for f in tqdm(files):
            data = load_image(f)
            label = [facet]

            X.append(data)
            y.append(label)

        X, y = unison_shuffled_copies(np.array(X), np.array(y))
        pivot = int(val_percent*len(X))
        X_res_val.extend(X[:pivot])
        y_res_val.extend(y[:pivot])
        X_res_train.extend(X[pivot:]) 
        y_res_train.extend(y[pivot:])

    return np.array(X_res_train),np.array(y_res_train), np.array(X_res_val),np.array(y_res_val) 

classes = ["airpixels",
    "fursty",
    "shortstache",
    "haakeaulana",
    "thiswildidea",
    "loki",
    "hannes_becker",
    "danielkordan",
    "cestmaria",
    "jessfindlay"]


train_path = "../Images/images"
X_train, y_train, X_valid, y_valid = load(train_path, classes)


train_x = X_train
train_y = MultiLabelBinarizer(classes).fit_transform(y_train)

valid_x = X_valid
valid_y = MultiLabelBinarizer(classes).fit_transform(y_valid)


train_h5_name = 'train_separate_10.h5'
valid_h5_name = 'validate_separate_10.h5'

with h5py.File(train_h5_name, 'w') as hf:
    hf.create_dataset("images",  data=train_x)
    hf.create_dataset("labels", data = train_y)

with h5py.File(valid_h5_name, 'w') as hf:
    hf.create_dataset("images",  data=valid_x)
    hf.create_dataset("labels", data = valid_y)

with h5py.File(train_h5_name, 'r') as hf:
        X = hf['images'][:]
        y = hf['labels'][:]


print('Preprocess done')
