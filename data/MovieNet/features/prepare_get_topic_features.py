import json
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from nltk.tokenize import word_tokenize
import random
import random
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import cn_clip.clip as cn_clip
import torch

import clip
def text_feature_save(ids, save_path, vector):
    for i in range(len(ids)):
        filename = os.path.join(save_path, ids[i])
        filename = filename + '.npy'
        document_vectors_array = np.array(vector[i])
        np.save(filename, document_vectors_array)
def replace_substring(string, replacement_string, replacement_percentage):
    random.seed(42)
    start_index = int(len(string) * random.random())
    substring_length = int(len(string) * replacement_percentage)
    replacement_substring = replacement_string[:substring_length]
    result = string[:start_index] + replacement_substring + string[start_index + substring_length:]
    return result


def get_text_augment_sim(ids, documents, sim_index, save_path, k):
    for i in range(len(ids)):
        tt_text_ori = documents[i]
        for j in range(k):
            tt_text_aug = documents[int(sim_index[i][j+1])]
            replacement_percentage = 0.15/k
            tt_text_ori = replace_substring(tt_text_ori, tt_text_aug, replacement_percentage)
        filename = os.path.join(save_path, ids[i])
        filename = filename +'.txt'
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(tt_text_ori)

def load_from_txt(ids, load_path):
    all_text = []
    for i in ids:
        tt_txt = load_path + '/' + str(i) + '.txt'
        if os.path.exists(tt_txt):
            with open(tt_txt, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    line = line.strip()
                    all_text.append(line)
        else:
            print('not exists')
    return all_text
def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data
def get_topic_clip_features(topic_texts, normalize=False):
    truncation_topic_texts = []
    truncation_num = 60
    for topic_t in topic_texts:
        #topic_t_re = topic_t.replace('_','')
        if len(topic_t) > truncation_num:
            topic_t_re = topic_t[0:truncation_num].replace('_', '')

            truncation_topic_texts.append(topic_t_re)
        else:
            topic_t_re = topic_t[0:truncation_num].replace('_', '')

            truncation_topic_texts.append(topic_t_re)

    # text_inputs = clip.tokenize(truncation_topic_texts).to(device)
    # #text_inputs = cn_clip.tokenize(topic_texts).to(device)
    # text_features = model.encode_text(text_inputs)
    # return text_features.cpu().detach().numpy()

    text_inputs = cn_clip.tokenize(topic_texts).to(device)
    text_features = cn_model.encode_text(text_inputs)
    return text_features.cpu().detach().numpy()
def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum()
if __name__ == '__main__':

    movie1k_json = ' /data/movieNet/files/movie1K.split.json'
    text_path = 'subtitle'
    with open(movie1k_json, 'r') as f:
        movie1K_split = json.load(f)
    id_all = movie1K_split['full']


    feature_root = "../features"
    save_file = 'topic_BERTopic.pkl'
    feature_folder = os.path.join(feature_root, "Topic_features")
    text_topic_path = ' /data/MovieNet1100/text_topic'
    #text_topic_pro_path = ' /second work/code/sec-5-25-2-L/data/Douban2595/text_topic_pro'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cn_model, cn_preprocess = cn_clip.load_from_name("ViT-L-14-336", device=device,
                                                     download_root=' /data/MovieNet1100/features/cache/checkpoints/cn_clip_cache')
    model, preprocess = clip.load("ViT-L/14@336px", device=device,
                                  download_root=' /data/MovieNet1100/features/cache/checkpoints/clip_cache')

    text_feat_dict = {}
    save_path = os.path.join(feature_folder, save_file)
    index = 0
    # with open(save_path, "wb") as handle:
    #     pkl.dump(text_feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    maxlen = 32
    for imdb in tqdm(id_all):
        #print(index)
        index = index + 1
        filename = os.path.join(text_topic_path, imdb)
        filename = filename + '.pkl'

        topic_pkl = read_pkl(filename)
        topic_texts = topic_pkl['topic']
        topic_pro = topic_pkl['pro']
        if len(topic_texts ) > maxlen:
            maxlen = len(topic_texts )
        topic_features = get_topic_clip_features(topic_texts)
        text_feat_dict[imdb] = {}

        padded_topic_features = np.pad(
            topic_features,
            pad_width=((0, maxlen - topic_features.shape[0]), (0, 0)),  
            mode='constant',  
            constant_values=0.0  
        )
        softtopic_pro = softmax(topic_pro)
        padded_topic_pro = np.pad(
            softtopic_pro,
            pad_width=(0, maxlen - topic_features.shape[0]),  
            mode='constant', 
            constant_values=0.0  
        )

        text_feat_dict[imdb]['topic'] = padded_topic_features
        text_feat_dict[imdb]['w'] = padded_topic_pro
        print(index)
    with open(save_path, "wb") as handle:
        pkl.dump(text_feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print(maxlen)





