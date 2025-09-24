import os
from os.path import join, basename, dirname
import pickle as pkl
import argparse
from tqdm import tqdm
import torch
from PIL import Image
import pandas as pd
import json
import numpy as np
import dgl
from dgl.nn import GraphConv
import dgl.function as fn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data
def get_g(movies):
    # 计算每对电影的共同元素数量
    edges_with_weight = {}
    movies_list = list(movies.items())

    for i, (m1, elems1) in enumerate(movies_list):
        for j, (m2, elems2) in enumerate(movies_list[i + 1:], i + 1):
            common = set(elems1) & set(elems2)
            if common:
                weight = len(common)
                # if weight >1:
                #     print(weight)
                edges_with_weight[(i, j)] = weight
                edges_with_weight[(j, i)] = weight  # 无向图
            else:
                edges_with_weight[(i, j)] = 0
                edges_with_weight[(j, i)] = 0

    # 转换为边列表和权重
    src, dst = zip(*edges_with_weight.keys()) if edges_with_weight else ([], [])
    weights = list(edges_with_weight.values())
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_list = scaler.fit_transform(np.array(weights).reshape(-1, 1)).flatten()
    return src, dst, normalized_list
def cacu_weights_v(feats):
    weights = []
    n = feats.shape[0]
    f_norm = normalize(feats, norm='l2', axis=1)
    cosine_sim = f_norm @ f_norm.T
    for i in range(n):
        for j in range(n):
            if i != j:
                weights.append(cosine_sim[i][j])
    return torch.tensor(weights, dtype=torch.float32)
def cacu_weights_t(feats, w_topic_feature, imdb_ids):
    feats_t = []
    w_t = []

    w_feats = []
    # for id in imdb_ids:
    #     feats_t.append(topic_feature[id]['topic'])
    #     w_t.append(topic_feature[id]['w'])
    for id in imdb_ids:
        w_feats.append(w_topic_feature[id])
    #feats_t_np = np.vstack(feats_t)
    # feats_t_np = np.array(feats_t)
    # w_t_np = np.vstack(w_t)
    # feats_t_weighted = np.sum(feats_t_np * w_t_np[:, :, np.newaxis], axis=1)

    w_feats_t_np = np.squeeze(np.array(w_feats), axis=1)
    #f_t_norm_1 = normalize(feats_t_weighted, norm='l2', axis=1)

    f_t_norm = normalize(w_feats_t_np, norm='l2', axis=1)
    #cosine_similarity(np.expand_dims(feats_t_weighted[0], axis=0) , w_topic_feature['tt0810988']) sim =1
    weights = []
    n = len(w_feats)
    cosine_sim = f_t_norm @ f_t_norm.T
    for i in range(n):
        for j in range(n):
            if i != j:
                weights.append(cosine_sim[i][j])
    return torch.tensor(weights, dtype=torch.float32), torch.tensor(w_feats_t_np)

re = 'softmax'
visual_feature_path_train = ' /data/ MovieNet/features/Visual_features_L/mean_ViT-L-14-336_train.pkl'
visual_feature_path_val = ' /data/ MovieNet/features/Visual_features_L/mean_ViT-L-14-336_val.pkl'
visual_feature_path_test = ' /data/ MovieNet/features/Visual_features_L/mean_ViT-L-14-336_test.pkl'
visual_feature_train = read_pkl(visual_feature_path_train)
visual_feature_val = read_pkl(visual_feature_path_val)
visual_feature_test = read_pkl(visual_feature_path_test)
visual_feature = visual_feature_train | visual_feature_val | visual_feature_test

topic_feature_path = ' /data/ MovieNet/features/Topic_features/topic_BERTopic_en.pkl'
topic_feature = read_pkl(topic_feature_path)

w_topic_feature_path = ' /data/ MovieNet/features/Topic_features/w_topic_BERTopic.pkl'
w_topic_feature = read_pkl(w_topic_feature_path)

meta_json_path_train = ' /data/ MovieNet/meta/shot_meta_train_L.json'
with open(meta_json_path_train, 'r') as handle:
    meta_json_train = json.load(handle)
meta_json_path_val = ' /data/ MovieNet/meta/shot_meta_val_L.json'
with open(meta_json_path_val, 'r') as handle:
    meta_json_val = json.load(handle)
meta_json_path_test = ' /data/ MovieNet/meta/shot_meta_test_L.json'
with open(meta_json_path_test, 'r') as handle:
    meta_json_test = json.load(handle)
meta_json = meta_json_train | meta_json_val | meta_json_test
imdb_ids = meta_json_train['id'] + meta_json_val['id'] + meta_json_test['id']
meta = meta_json_train['meta'] + meta_json_val['meta'] + meta_json_test['meta']
movies = {}
for id, cast in zip(imdb_ids, meta):
    movies[id] = cast
src, dst, weights = get_g(movies)

feats = []
for id in imdb_ids:
    feats.append(visual_feature[id])
feats_np = np.vstack(feats)
#+++

#feats_t_np = np.vstack(feats_t)

weights_v = cacu_weights_v(feats_np)
weights_t, feats_t = cacu_weights_t(topic_feature, w_topic_feature, imdb_ids)

l2_norm = torch.norm(weights_v, p=2)
weights_v_l2 = weights_v / (l2_norm + 1e-8)

l1_norm = torch.sum(torch.abs(weights_v))  # 计算 L1 范数
weights_v_l1 = weights_v / (l1_norm + 1e-8)

l2_norm = torch.norm(weights_t, p=2)
weights_t_l2 = weights_t / (l2_norm + 1e-8)

l1_norm = torch.sum(torch.abs(weights_t))  
weights_t_l1 = weights_t / (l1_norm + 1e-8)

feats_g = torch.tensor(feats_np).squeeze(1)
g = dgl.graph((src, dst))
g.ndata['feat'] =feats_g.float()
#total = sum(weights)
#weights = [x / total for x in weights]
weights_g = torch.tensor(weights, dtype=torch.float32)
#g.edata['weight'] = weights_g
l2_norm = torch.norm(weights_g, p=2)
weights_m_l2 = weights_g / (l2_norm + 1e-8)
#weights_a = torch.softmax(weights_g,dim=0)

l1_norm = torch.sum(torch.abs(weights_g))  
weights_m_l1 = weights_g / (l1_norm + 1e-8)

g.edata['w'] = weights_m_l2/3 + weights_v_l2/3 + weights_t_l2/3 # edge_weight / edge_weight.sum() torch.sort(torch.softmax(weights_g,dim=0), descending=True)
gcn_layer = GraphConv(768, 768, weight=False, bias=False, norm='none', allow_zero_in_degree=True)

out = gcn_layer(g, feats_g, edge_weight=g.edata['w'])
out = out + feats_g
#cosine_similarity(out,feats_g)
out = out.numpy()
dic_train = {}
dic_val = {}
dic_test = {}
index = 0
for i in range(len(imdb_ids)):
    if i <len(meta_json_train['id']):
        dic_train[imdb_ids[i]] = np.expand_dims(out[i], axis=0)
    elif i >= len(meta_json_train['id']) and i < len(meta_json_train['id']) + len(meta_json_val['id']):
        dic_val[imdb_ids[i]] = np.expand_dims(out[i], axis=0)
    elif i >=len(meta_json_train['id']) + len(meta_json_val['id']) and i <len(meta_json_train['id']) + len(meta_json_val['id']) + len(meta_json_test['id']) :
        dic_test[imdb_ids[i]] = np.expand_dims(out[i], axis=0)
    else:
        print('error %d' % (i))
print('end')
visual_feature_path_train = ' /data/ MovieNet/features/Visual_features_L/VTM_ViT-L-14-336_train.pkl'
visual_feature_path_val = ' /data/ MovieNet/features/Visual_features_L/VTM_ViT-L-14-336_val.pkl'
visual_feature_path_test = ' /data/ MovieNet/features/Visual_features_L/VTM_ViT-L-14-336_test.pkl'
with open(visual_feature_path_train, "wb") as handle:
    pkl.dump(dic_train, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open(visual_feature_path_val, "wb") as handle:
    pkl.dump(dic_val, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open(visual_feature_path_test, "wb") as handle:
    pkl.dump(dic_test, handle, protocol=pkl.HIGHEST_PROTOCOL)

g_t = dgl.graph((src, dst))
g_t.ndata['feat'] =feats_t.float()
g_t.edata['w'] = weights_m_l2/3 + weights_v_l2/3 + weights_t_l2/3 # edge_weight / edge_weight.sum() torch.sort(torch.softmax(weights_g,dim=0), descending=True)
out_t = gcn_layer(g_t, feats_t.float(), edge_weight=g.edata['w'])
out_t = out_t + feats_t

dic_t = {}
for i in range(len(imdb_ids)):
    dic_t[imdb_ids[i]] = np.expand_dims(out_t[i], axis=0)
t_feature_path_train = ' /data/ MovieNet/features/Topic_features/VTM_topic_BERTopic.pkl'

with open(t_feature_path_train, "wb") as handle:
    pkl.dump(dic_t, handle, protocol=pkl.HIGHEST_PROTOCOL)
