import os
import json
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import re
from tqdm import tqdm

def get_visiual_augment_sim(sim_score, visual_path, train_inf_path, val_inf_path, test_inf_path):
    aug_vision_path = 'aug_vision'
    with open(train_inf_path, 'r') as f:
        json_train = json.load(f)
    ids_train = json_train['id']
    shot_select_train = json_train['shot_select']
    shot_augment_select_train = json_train['shot_augment_select']

    with open(val_inf_path, 'r') as f:
        json_val = json.load(f)
    ids_val = json_val['id']
    shot_select_val = json_val['shot_select']
    shot_augment_select_val = json_val['shot_augment_select']

    with open(test_inf_path, 'r') as f:
        json_test = json.load(f)
    ids_test = json_test['id']
    shot_select_test = json_test['shot_select']
    shot_augment_select_test= json_test['shot_augment_select']

    ids_all = ids_train + ids_val + ids_test
    shot_select_all = shot_select_train + shot_select_val + shot_select_test
    shot_augment_select_all = shot_augment_select_train + shot_augment_select_val + shot_augment_select_test
    visual_all = []
    visual_neighbor_all = []
    width = 0
    height = 0
    for i in range(len(sim_score)):
        tt_path = os.path.join(visual_path, ids_all[i])
        for j in shot_select_all[i]:
            shot_select = os.path.join(tt_path, j)
            img = Image.open(shot_select)
            width = max(width, img.width)
            height = max(height, img.height)
            img.close()

    for i in tqdm(range(len(sim_score))):
        tt_path = os.path.join(visual_path, ids_all[i])
        visual_tt = []
        visual_neighbor_tt = []
        for j in shot_select_all[i]:
            shot_select = os.path.join(tt_path, j)
            img = Image.open(shot_select)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_tt.append(img_array)
            img.close()
            shot_select_1 = shot_select.replace('0.jpg', '1.jpg')
            img = Image.open(shot_select_1)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_tt.append(img_array)
            img.close()
            shot_select_2 = shot_select.replace('0.jpg', '2.jpg')
            img = Image.open(shot_select_2)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_tt.append(img_array)
            img.close()
        visual_all.append(visual_tt)
        for j in shot_augment_select_all[i]:
            neighbor = []
            for k in j :
                shot_select_neighbor = os.path.join(tt_path, k)
                img = Image.open(shot_select_neighbor)
                img = img.resize((width, height))
                img_array = np.array(img)
                neighbor.append(img_array)
                img.close()
            visual_neighbor_tt.append(neighbor)
            neighbor = []
            for k in j :
                shot_select_neighbor_1 = shot_select_neighbor.replace('0.jpg', '1.jpg')
                img = Image.open(shot_select_neighbor_1)
                img = img.resize((width, height))
                img_array = np.array(img)
                neighbor.append(img_array)
                img.close()
            visual_neighbor_tt.append(neighbor)
            neighbor = []
            for k in j :
                shot_select_neighbor_2 = shot_select_neighbor.replace('0.jpg', '2.jpg')
                img = Image.open(shot_select_neighbor_2)
                img = img.resize((width, height))
                img_array = np.array(img)
                neighbor.append(img_array)
                img.close()
            visual_neighbor_tt.append(neighbor)
        visual_neighbor_all.append(visual_neighbor_tt)
    if not os.path.exists(aug_vision_path):
        os.mkdir(aug_vision_path)
    for i in tqdm(range(len(sim_score))):
        tt_path = os.path.join(aug_vision_path, ids_all[i])

        array_tt = visual_all[i]

        array_neighbor = visual_neighbor_all[i]

        sim_index = sim_score[i]

        for j in range(0,len(array_tt),3):
            array1 = array_tt[j]
            array2 = array_neighbor[j]
            array_sim = visual_all[sim_index][j]
            visual2 = np.sum(array2, axis=0)

            visual2 = visual2 / len(array2)
            visual = 0.7 * array1 + 0.1 * array_sim + 0.2 * visual2
            if not os.path.exists(tt_path):
                os.makedirs(tt_path)
            filename = os.path.join(tt_path, shot_select_all[i][int(j/3)])
            blended_image = Image.fromarray(visual.astype(np.uint8))
            blended_image.save(filename)
            array1 = array_tt[j+1]
            array2 = array_neighbor[j+1]
            array_sim = visual_all[sim_index][j+1]
            visual2 = np.sum(array2, axis=0)

            visual2 = visual2 / len(array2)
            visual = 0.7 * array1 + 0.1 * array_sim + 0.2 * visual2
            if not os.path.exists(tt_path):
                os.makedirs(tt_path)
            filename = os.path.join(tt_path, shot_select_all[i][int(j/3)])
            filename1 = filename.replace('0.jpg', '1.jpg')
            blended_image = Image.fromarray(visual.astype(np.uint8))
            blended_image.save(filename1)
            array1 = array_tt[j+2]
            array2 = array_neighbor[j+2]
            array_sim = visual_all[sim_index][j+2]
            visual2 = np.sum(array2, axis=0)
            visual2 = visual2 / len(array2)
            visual = 0.7 * array1 + 0.1 * array_sim + 0.2 * visual2
            if not os.path.exists(tt_path):
                os.makedirs(tt_path)
            filename = os.path.join(tt_path, shot_select_all[i][int(j/3)])
            filename2 = filename.replace('0.jpg', '2.jpg')
            blended_image = Image.fromarray(visual.astype(np.uint8))
            blended_image.save(filename2)
def get_sim(path, ids):
    n = len(ids)
    score = [[0] * n for _ in range(n)]
    meta = []
    score_sim = []
    for id in ids:
        tt_path = os.path.join(path, str(id) + '.json')
        tt_dic = {}
        with open(tt_path, 'r') as f:
            tt_meta = json.load(f)
            tt_country = tt_meta['country']
            tt_version = int(tt_meta['version'][0]['runtime'].replace('min',''))
            tt_runtime = -1
            if tt_version <60:
                tt_runtime = 0
            elif tt_version >=60 and tt_version <90:
                tt_runtime = 1
            elif tt_version >=90 and tt_version <120:
                tt_runtime = 2
            elif tt_version >=120:
                tt_runtime = 3
            else:
                print(str(tt_runtime))
                print('wrong')
            if tt_runtime == -1:
                print('runtime is wrong')
            tt_dic['runtime'] = tt_runtime
            tt_dic['country'] = tt_country
        meta.append(tt_dic)
    for i in range(n):
        for j in range(n):
            if i == j:
                score[i][j] = -1
            else:
                score_country = 1 if meta[i]['country'] == meta[j]['country'] else 0
                score_runtime = 1 if meta[i]['runtime'] == meta[j]['runtime'] else 0
                score[i][j] = score_country + score_runtime
    for i in score:
        index = i.index(max(i))
        score_sim.append(index)
    return score_sim

def get_visiual_augment_sim_p(sim_score, visual_path, topk, train_inf_path, val_inf_path, test_inf_path):
    aug_vision_path = 'aug_vision'
    with open(train_inf_path, 'r') as f:
        json_train = json.load(f)
    ids_train = json_train['id']
    shot_select_train = json_train['shot_select']
    shot_augment_select_train = json_train['shot_augment_select']

    with open(val_inf_path, 'r') as f:
        json_val = json.load(f)
    ids_val = json_val['id']
    shot_select_val = json_val['shot_select']
    shot_augment_select_val = json_val['shot_augment_select']

    with open(test_inf_path, 'r') as f:
        json_test = json.load(f)
    ids_test = json_test['id']
    shot_select_test = json_test['shot_select']
    shot_augment_select_test= json_test['shot_augment_select']

    ids_all = ids_train + ids_val + ids_test
    shot_select_all = shot_select_train + shot_select_val + shot_select_test
    shot_augment_select_all = shot_augment_select_train + shot_augment_select_val + shot_augment_select_test

    width = 0
    height = 0
    for i in range(len(sim_score)):
        tt_path = os.path.join(visual_path, ids_all[i])
        for j in shot_select_all[i]:
            shot_select = os.path.join(tt_path, j)
            img = Image.open(shot_select)
            width = max(width, img.width)
            height = max(height, img.height)
            img.close()

    for i in tqdm(range(len(sim_score))):
        tt_path = os.path.join(visual_path, ids_all[i])
        visual_all_i = []
        for j in shot_select_all[i]:
            shot_select = os.path.join(tt_path, j)
            img = Image.open(shot_select)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_all_i.append(img_array)
            img.close()

            shot_select_1 = shot_select.replace('0.jpg', '1.jpg')
            if not os.path.exists(shot_select_1):
                shot_select_1 = shot_select
            img = Image.open(shot_select_1)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_all_i.append(img_array)
            img.close()
            shot_select_2 = shot_select.replace('0.jpg', '2.jpg')
            if not os.path.exists(shot_select_2):
                shot_select_2 = shot_select
            img = Image.open(shot_select_2)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_all_i.append(img_array)
            img.close()

        visual_neighbor_all_i = []
        for j in shot_augment_select_all[i]:
            neighbor = []
            for k in j :
                shot_select_neighbor = os.path.join(tt_path, k)
                img = Image.open(shot_select_neighbor)
                img = img.resize((width, height))
                img_array = np.array(img)
                neighbor.append(img_array)
                img.close()

            visual_neighbor_all_i.append(neighbor)
            neighbor = []
            for k in j :
                shot_select_neighbor_1 = shot_select_neighbor.replace('0.jpg', '1.jpg')
                if not os.path.exists(shot_select_neighbor_1):
                    shot_select_neighbor_1 = shot_select_neighbor
                img = Image.open(shot_select_neighbor_1)
                img = img.resize((width, height))
                img_array = np.array(img)
                neighbor.append(img_array)
                img.close()
            visual_neighbor_all_i.append(neighbor)
            neighbor = []
            for k in j :
                shot_select_neighbor_2 = shot_select_neighbor.replace('0.jpg', '2.jpg')
                if not os.path.exists(shot_select_neighbor_2):
                    shot_select_neighbor_2 = shot_select_neighbor
                img = Image.open(shot_select_neighbor_2)
                img = img.resize((width, height))
                img_array = np.array(img)
                neighbor.append(img_array)
                img.close()
            visual_neighbor_all_i.append(neighbor)
        #sim
        sim_index = sim_score[i]
        visual_all_sim = []
        sim_tt_path = os.path.join(visual_path, ids_all[sim_index])
        for j in shot_select_all[sim_index]:
            shot_select = os.path.join(sim_tt_path, j)
            img = Image.open(shot_select)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_all_sim.append(img_array)
            img.close()

            shot_select_1 = shot_select.replace('0.jpg', '1.jpg')
            if not os.path.exists(shot_select_1):
                shot_select_1 = shot_select
            img = Image.open(shot_select_1)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_all_sim.append(img_array)
            img.close()
            shot_select_2 = shot_select.replace('0.jpg', '2.jpg')
            if not os.path.exists(shot_select_2):
                shot_select_2 = shot_select
            img = Image.open(shot_select_2)
            img = img.resize((width, height))
            img_array = np.array(img)
            visual_all_sim.append(img_array)
            img.close()

        tt_path = os.path.join(aug_vision_path, ids_all[i])
        for j in range(0,len(visual_all_i),3):
            array1 = visual_all_i[j]
            array2 = visual_neighbor_all_i[j]
            array_sim = visual_all_sim[j]
            if len(array2)!=topk+1:
                visual2 = 0
            else:
                visual2 = np.sum(array2, axis=0)
                visual2 = visual2 / len(array2)
            visual = 0.7 * array1 + 0.1 * array_sim + 0.2 * visual2
            if not os.path.exists(tt_path):
                os.makedirs(tt_path)
            filename = os.path.join(tt_path, shot_select_all[i][int(j/3)])
            blended_image = Image.fromarray(visual.astype(np.uint8))
            blended_image.save(filename)
            array1 = visual_all_i[j+1]
            array2 = visual_neighbor_all_i[j+1]
            array2 = np.nan_to_num(array2, nan=0.0)
            array_sim = visual_all_sim[j+1]
            if len(array2)!=topk+1:
                visual2 = 0
            else:
                visual2 = np.sum(array2, axis=0)
                visual2 = visual2 / len(array2)
            visual = 0.7 * array1 + 0.1 * array_sim + 0.2 * visual2
            if not os.path.exists(tt_path):
                os.makedirs(tt_path)
            filename = os.path.join(tt_path, shot_select_all[i][int(j/3)])
            filename1 = filename.replace('0.jpg', '1.jpg')
            blended_image = Image.fromarray(visual.astype(np.uint8))
            blended_image.save(filename1)
            array1 = visual_all_i[j+2]
            array2 = visual_neighbor_all_i[j+2]
            array2 = np.nan_to_num(array2, nan=0.0)
            array_sim = visual_all_sim[j+2]
            if len(array2)!=topk+1:
                visual2 = 0
            else:
                visual2 = np.sum(array2, axis=0)
                visual2 = visual2 / len(array2)
            visual = 0.7 * array1 + 0.1 * array_sim + 0.2 * visual2
            if not os.path.exists(tt_path):
                os.makedirs(tt_path)
            filename = os.path.join(tt_path, shot_select_all[i][int(j/3)])
            filename2 = filename.replace('0.jpg', '2.jpg')
            blended_image = Image.fromarray(visual.astype(np.uint8))
            blended_image.save(filename2)
        del visual_neighbor_all_i
        del visual_all_i
if __name__ == '__main__':
    topk = 1
    movie1k_json='/data/movieNet/files/movie1K.split.json'
    data_path = '/data/movieNet'
    split_path = 'split'
    shotdata_path = os.path.join(data_path, '240P/all')
    metadata_path = os.path.join(data_path, 'files/meta')
    train_inf_path = 'split/train_inf_L.json'
    val_inf_path = 'split/val_inf_L.json'
    test_inf_path = 'split/test_inf_L.json'
    with open(movie1k_json, 'r') as f:
        movie1K_split = json.load(f)
    id_all = movie1K_split['full']
    sim_score = get_sim(metadata_path, id_all)
    get_visiual_augment_sim_p(sim_score, shotdata_path, topk, train_inf_path, val_inf_path, test_inf_path)






