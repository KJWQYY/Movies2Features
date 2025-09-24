import json
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from nltk.tokenize import word_tokenize
import random
import random
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
import re
import pickle as pkl
from tqdm import tqdm

def text_filter(textfile):
    all_text = []
    all_text_filter = []
    all_text_result = []
    seg_text = []
    with open(textfile, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            seg_text.append(line)
            if line == '':
                all_text.append(seg_text)
                seg_text = []

    for i in all_text:
        text = i[2:-1]
        if len(text) > 0:
            all_text_filter.append(text)
    one = ''
    for i in all_text_filter:
        for j in i:
            one = one + j
    if len(one) > 0:
        for i in all_text_filter:
            for j in i:
                all_text_result.append(j)
        return one
    else:
        print(textfile)
        return one


def write_pkl(path: str, data: dict):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    return 1
def text_save(ids, save_path, dic):
    filename = os.path.join(save_path, ids)
    filename = filename + '.pkl'
    write_pkl(filename, dic)

if __name__ == '__main__':
    movie1k_json = 'G:/data/movieNet/files/movie1K.split.json'
    text_path = 'subtitle'
    with open(movie1k_json, 'r') as f:
        movie1K_split = json.load(f)
    id_all = movie1K_split['full']
    document_dic = {}
    document_list = []
    exist = []
    for i in tqdm(id_all):
        if os.path.exists(os.path.join(text_path,i+'.srt')):
            document = text_filter(os.path.join(text_path,i+'.srt'))
            if len(document) > 0:
                document_list.append(document)
                document_dic[i] = document
                exist.append(i)
            else:
                document = 'no text'
                document_list.append(document)
                document_dic[i] = document
        else:
            document = 'no text'
            document_list.append(document)
            document_dic[i] = document

    k = 2
    text_topic_path = '/data/MovieNet1100/text_topic'
    text_topic_pro_path = '/data/MovieNet1100/text_topic_pro'

    #document_list = document_list[0:1]
    #id_all = id_all[0:1]
    index = 0
    for doc, imdb in tqdm(zip(document_list, id_all)):
        print(index)
        index = index + 1
        filename = os.path.join(text_topic_path, imdb)
        filename = filename + '.pkl'

        filename_pro = os.path.join(text_topic_pro_path, imdb)
        filename_pro = filename_pro + '.pkl'


        #raw_sentences  = re.split(r'([。！？…\n])', doc)
        raw_sentences = re.split(r'([.!?。！？…\n])', doc)
        sentences = []
        buffer = ""
        min_length = 50
        #min_length = 10
        for i, s in enumerate(raw_sentences):
            buffer += s
            if len(buffer) >= min_length or s == '\n':
                sentences.append(buffer.strip())
                buffer = ""
            if i == len(raw_sentences) - 1:
                if len(buffer) < min_length:
                    if len(sentences) > 0:

                        sentences[-1] += buffer
                    else:
                        sentences.append(buffer)


        max_lens = 0
        for sen in sentences:
            if len(sen) > max_lens:
                max_lens = len(sen)
        #print(max_lens)
        top_s = 10
        dic = {}
        dic['topic']  = []
        dic['pro'] = []
        dic_ = {}
        dic_['topic'] = []

        if len(sentences) < 11:

            text_topic = ''
            pro = 0.0
            for sen in sentences:
                text_topic = text_topic + sen+ '_'
            dic['topic'].append(sen)
            dic['pro'].append(1/len(sentences)) #
            dic_['topic']= None
        else:
            #continue
            topic_model = BERTopic(language="multilingual")
            topics, probs = topic_model.fit_transform(sentences)

            for topic_num in topic_model.get_topic_info()["Topic"]:
                text_topic = ''
                pro = 0.0
                topic_i = topic_model.get_topic(topic_num)# 10 * 2
                dic_['topic'].append(topic_i)
                for topic_tuple in topic_i:
                    text_topic = text_topic + topic_tuple[0] + '_'
                    pro = pro + topic_tuple[1]

                dic['topic'].append(text_topic)
                #print(text_topic)
                dic['pro'].append(pro)

        text_save(imdb, text_topic_path, dic)
        text_save(imdb, text_topic_pro_path, dic_)







