import json
import os
from tqdm import tqdm
def get_shot_idx(path,ids,sampling_num, window):
    shot_select = []
    shot_augment_select = []
    for id in tqdm(ids):
        tt_path = os.path.join(path, id)
        img_names = os.listdir(tt_path)
        shot_idx_list = []
        for img in img_names:
            imgname_split = img.split('_')
            shot_idx = str(imgname_split[1])
            if not shot_idx in shot_idx_list:
                shot_idx_list.append(shot_idx)
        maxid = max(shot_idx_list, key=int)
        shot_num = len(shot_idx_list)
        step = int(shot_num/(sampling_num+1))
        tt_shot_select_idx_list = []
        tt_shot_augment_select_idx_list = []

        for i in range(sampling_num):
            tt_shot_select_idx_list.append('shot_'+str(shot_idx_list[(i+1)*step])+'_img_0.jpg')
            window_shot_augment_select_idx_list = []

            for j in range(0,window):
                if ( (i+1)*step + j +1) < int(maxid):
                    window_shot_augment_select_idx_list.append('shot_'+str(shot_idx_list[(i+1)*step + j + 1])+'_img_0.jpg')
            tt_shot_augment_select_idx_list.append(window_shot_augment_select_idx_list)
        shot_select.append(tt_shot_select_idx_list)
        shot_augment_select.append(tt_shot_augment_select_idx_list)
    result_dic = {}
    result_dic['id'] = ids
    result_dic['shot_select'] = shot_select
    result_dic['shot_augment_select'] = shot_augment_select

    return result_dic
if __name__ == '__main__':
    movie1k_json='G:/data/movieNet/files/movie1K.split.json'
    data_path = 'G:/data/movieNet'
    split_path = 'split'
    shotdata_path = os.path.join(data_path, '240P/all')
    with open(movie1k_json, 'r') as f:
        movie1K_split = json.load(f)
    train_id = movie1K_split['train']
    val_id = movie1K_split['val']
    test_id = movie1K_split['test']
    id_all = movie1K_split['full']
    sampling_num = 8
    window = 2
    shot_select_train = get_shot_idx(shotdata_path,train_id,sampling_num, window)
    shot_select_val = get_shot_idx(shotdata_path, val_id, sampling_num, window)
    shot_select_test = get_shot_idx(shotdata_path, test_id, sampling_num, window)
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    train_inf_path = os.path.join(split_path, 'train_inf_L.json')
    val_inf_path = os.path.join(split_path, 'val_inf_L.json')
    test_inf_path = os.path.join(split_path, 'test_inf_L.json')

    with open(train_inf_path, "w") as json_file:
        json.dump(shot_select_train, json_file)
    with open(val_inf_path, "w") as json_file:
        json.dump(shot_select_val, json_file)
    with open(test_inf_path, "w") as json_file:
        json.dump(shot_select_test, json_file)
