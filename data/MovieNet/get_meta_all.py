import json
import os
def get_info(ids, anno_path, shot_select):
    shots_metainfo = []

    for i in range(len(ids)):
        tt_path = os.path.join(anno_path, str(ids[i]) + '.json')
        with open(tt_path, 'r') as f:
            tt_meta = json.load(f)
        imdb_id = tt_meta['imdb_id']
        cast = tt_meta['cast']
        shot_meta = []
        if cast == None:
            for j in shot_select[i]:
                shot_meta.append('unknow')
            shots_metainfo.append(shot_meta)
            continue
        for j in shot_select[i]:
            shot_idx = int(j.split('_')[1])
            find = 0
            for i in cast:
                shot_idx_temp = i['shot_idx']
                if shot_idx == shot_idx_temp:
                    pid = i['pid']
                    if pid == None:
                        pid = 'unknow'
                    shot_meta.append(pid)
                    find = 1
                    break
            if find == 0:
                shot_meta.append('unknow')
        shots_metainfo.append(shot_meta)
    return shots_metainfo
def get_metainfo(ids, meta_path, k):
    shots_metainfo = []

    for i in range(len(ids)):
        tt_path = os.path.join(meta_path, str(ids[i]) + '.json')
        with open(tt_path, 'r') as f:
            tt_meta = json.load(f)

        cast = tt_meta['cast']
        #character_id = cast['id']

        shot_meta = []
        if cast == None:
            shot_meta.append('unknow')
        else:
            # if len(cast) >= k:
            #     cast = cast[0:k]
            for ca in cast:
                character_id = ca['id']
                shot_meta.append(character_id)
        # if len(shot_meta) < k:
        #     num = k - len(shot_meta)
        #     for j in range(num):
        #         shot_meta.append('unknow')

        shots_metainfo.append(shot_meta)
    return shots_metainfo
def get_sim(anno, meta):
    re = []
    for i in range(0,len(anno)):
        same_num = 0
        for j in anno[i]:
            if j != 'unknow' and j != 'others':
                if j in meta[i]:
                    same_num = same_num + 1
        re.append(same_num / len(meta[i]))
    return re
if __name__ == '__main__':
    k1 = 5
    k2 = 10
    data_path = '/data/movieNet'
    anno_path = os.path.join(data_path, 'files/annotation')
    meta_path = os.path.join(data_path, 'files/meta')
    split_path = 'split'
    sava_path = 'meta'
    if not os.path.exists(sava_path):
        os.mkdir(sava_path)
    train_inf_path = os.path.join(split_path, 'train_inf_L.json')
    val_inf_path = os.path.join(split_path, 'val_inf_L.json')
    test_inf_path = os.path.join(split_path, 'test_inf_L.json')
    with open(train_inf_path, 'r') as f:
        json_train = json.load(f)
    with open(val_inf_path, 'r') as f:
        json_val = json.load(f)
    with open(test_inf_path, 'r') as f:
        json_test = json.load(f)

    ids_train = json_train['id']
    shot_select_train = json_train['shot_select']

    ids_val = json_val['id']
    shot_select_val = json_val['shot_select']

    ids_test = json_test['id']
    shot_select_test = json_test['shot_select']

    anno_train = get_info(ids_train, anno_path, shot_select_train)
    anno_val = get_info(ids_val, anno_path, shot_select_val)
    anno_test = get_info(ids_test, anno_path, shot_select_test)

    meta_train = get_metainfo(ids_train, meta_path, k2)
    meta_val = get_metainfo(ids_val, meta_path, k2)
    meta_test = get_metainfo(ids_test, meta_path, k2)

    json_train = {}
    json_val = {}
    json_test = {}
    json_train['id'] = ids_train
    json_val['id'] = ids_val
    json_test['id'] = ids_test
    json_train['anno'] = anno_train
    json_val['anno'] = anno_val
    json_test['anno'] = anno_test
    json_train['meta'] = meta_train
    json_val['meta'] = meta_val
    json_test['meta'] = meta_test

    # json_train['sim'] = get_sim(anno_train, meta_train)
    # json_val['sim'] = get_sim(anno_val, meta_val)
    # json_test['sim'] = get_sim(anno_test, meta_test)


    train_json = os.path.join(sava_path, 'shot_meta_train_L.json')
    val_json = os.path.join(sava_path, 'shot_meta_val_L.json')
    test_json = os.path.join(sava_path, 'shot_meta_test_L.json')
    with open(train_json, "w",encoding='utf-8') as json_file:
        json.dump(json_train, json_file)
    with open(val_json, "w",encoding='utf-8') as json_file:
        json.dump(json_val, json_file)
    with open(test_json, "w",encoding='utf-8') as json_file:
        json.dump(json_test, json_file)




