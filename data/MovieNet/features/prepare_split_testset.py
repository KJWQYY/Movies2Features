import os,glob
import json
import random
from tqdm import tqdm
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="../features", help='path to features')
parser.add_argument('--data_path', type=str, default="/data/movieNet", help='path to data')
parser.add_argument('--labelpath', type=str, default="../label", help='path to label')
parser.add_argument('--train_inf_path', type=str, default="../split/train_inf_L.json", help='path to train_inf_path')
parser.add_argument('--val_inf_path', type=str, default="../split/val_inf_L.json", help='path to val_inf_path')
parser.add_argument('--test_inf_path', type=str, default="../split/test_inf_L.json", help='path to test_inf_path')
parser.add_argument('--shotdata_path', type=str, default="/data/movieNet/240P/all", help='path to shot')
args = parser.parse_args()

def getClipFormat(line):
    line = line.split('\t')
    id = line[0]
    info = line[1]
    label = line[2]
    shot = str(info.split('/')[-1][5:9])
    f1 = info[:-5]+"0.jpg"
    f2 = info[:-5]+"1.jpg"
    f3 = info[:-5]+"2.jpg"
    return id,shot,f1,f2,f3,label
def getRawFrame(ids, shots ,uponpath, labelpath):
    res = []
    min_num = 1000
    for i in range(len(ids)):
        #line = ''
        tt_label_path = os.path.join(labelpath, ids[i]) +'.npy'
        tt_label = np.load(tt_label_path)
        label_index = np.where(tt_label == 1)[0]
        tt_label_str = ''
        if len(label_index) >= 1:
            if len(label_index) < min_num:
                min_num = len(label_index)
            tt_label_str += str(label_index[0])
            for j in range(len(label_index)-1):
                tt_label_str += ' ' + str(label_index[j+1])
        for shot in shots[i]:
            tt_id = os.path.join(uponpath, str(ids[i]))
            tt_shot_path = os.path.join(tt_id, str(shot))
            line = ids[i] + '\t' + tt_shot_path + '\t' + tt_label_str
            line = line.replace('movieNetZero', 'movieNet')
            line = line.replace('\\', '/')
            res.append(line)
        # rawFrame format

    return res

with open(args.train_inf_path, 'r') as f:
    json_train = json.load(f)
with open(args.val_inf_path, 'r') as f:
    json_val = json.load(f)
with open(args.test_inf_path, 'r') as f:
    json_test = json.load(f)
train_id = json_train['id']
val_id = json_val['id']
test_id = json_test['id']
train_shot = json_train['shot_select']
val_shot = json_val['shot_select']
test_shot = json_test['shot_select']

train_annotation = getRawFrame(train_id, train_shot, args.shotdata_path, args.labelpath)
val_annotation = getRawFrame(val_id, val_shot, args.shotdata_path, args.labelpath)
test_annotation = getRawFrame(test_id, test_shot, args.shotdata_path, args.labelpath)
all_annotation = train_annotation + val_annotation + test_annotation

train_txt = 'key_frame_train_L.txt'
val_txt = 'key_frame_val_L.txt'
test_txt = 'key_frame_test_L.txt'

with open(os.path.join(args.data_dir,train_txt), "w") as file:
    for i in train_annotation:
        file.write(i)
        file.write('\n')
with open(os.path.join(args.data_dir,val_txt), "w") as file:
    for i in val_annotation:
        file.write(i)
        file.write('\n')
with open(os.path.join(args.data_dir,test_txt), "w") as file:
    for i in test_annotation:
        file.write(i)
        file.write('\n')

train_res = {}
for i in train_annotation:
    id,shot,f1,f2,f3,label = getClipFormat(i)
    if id not in train_res.keys():
        train_res[id] = {}
    if not shot in train_res[id].keys():
        train_res[id][shot] = []
    if not label in train_res[id].keys():
        train_res[id]['label'] = label
    train_res[id][shot].append(f1)
    train_res[id][shot].append(f2)
    train_res[id][shot].append(f3)
val_res = {}
for i in val_annotation:
    id,shot,f1,f2,f3,label = getClipFormat(i)
    if id not in val_res.keys():
        val_res[id] = {}
    if not shot in val_res[id].keys():
        val_res[id][shot] = []
    if not label in val_res[id].keys():
        val_res[id]['label'] = label
    val_res[id][shot].append(f1)
    val_res[id][shot].append(f2)
    val_res[id][shot].append(f3)
test_res = {}
for i in test_annotation:
    id,shot,f1,f2,f3,label = getClipFormat(i)
    if id not in test_res.keys():
        test_res[id] = {}
    if not shot in test_res[id].keys():
        test_res[id][shot] = []
    if not label in test_res[id].keys():
        test_res[id]['label'] = label
    test_res[id][shot].append(f1)
    test_res[id][shot].append(f2)
    test_res[id][shot].append(f3)
with open(os.path.join(args.data_dir, "data_train_L.json"), 'w') as fv:
    json.dump(train_res, fv)
with open(os.path.join(args.data_dir, "data_val_L.json"), 'w') as fv:
    json.dump(val_res, fv)
with open(os.path.join(args.data_dir, "data_test_L.json"), 'w') as fv:
    json.dump(test_res, fv)
