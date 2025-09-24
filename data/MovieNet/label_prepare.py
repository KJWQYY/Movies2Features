import json
import os
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def get_multlabel(path,ids,labeldic):
    label_list = []
    label_temp = {}
    for id in ids:
        tt_path = os.path.join(path, str(id)+'.json')
        tt_label_list = []
        with open(tt_path, 'r') as f:
            tt_meta = json.load(f)
            tt_genres = tt_meta['genres']
            for genre in tt_genres:
                if not genre in labeldic:
                    len_dic = len(labeldic)
                    labeldic[genre] = len_dic

                if not genre in label_temp:

                    label_temp[genre] = 1
                else:
                    label_temp[genre] = label_temp[genre] + 1

                    #label_temp[genre] = label_temp[genre] + 1
                #if genre != 'Documentary':
                tt_label_list.append(labeldic[genre])
        label_list.append(tt_label_list)
    return label_list, label_temp
def get_multlabel_fromdic(path,ids,labeldic):
    label_list = []
    label_temp = {}
    for id in ids:
        tt_path = os.path.join(path, str(id)+'.json')
        tt_label_list = []
        with open(tt_path, 'r') as f:
            tt_meta = json.load(f)
            tt_genres = tt_meta['genres']
            for genre in tt_genres:
                # if not genre in labeldic:
                #     len_dic = len(labeldic)
                #     labeldic[genre] = len_dic

                if not genre in label_temp:

                    label_temp[genre] = 1
                else:
                    label_temp[genre] = label_temp[genre] + 1


                tt_label_list.append(labeldic[genre])
        label_list.append(tt_label_list)
    return label_list, label_temp
if __name__ == '__main__':
    #movieNet 1k split

    movie1k_json='/data/movieNet/files/movie1K.split.json'
    data_path = '/data/movieNet'
    metadata_path = os.path.join(data_path, 'files/meta')
    with open(movie1k_json, 'r') as f:
        movie1K_split = json.load(f)
    train_id = movie1K_split['train']
    val_id = movie1K_split['val']
    test_id = movie1K_split['test']
    id_all = movie1K_split['full']

    label_path = 'label'
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    #get label
    labeldic = {}
    label_list_train, label_temp_train = get_multlabel(metadata_path, train_id, labeldic)
    label_list_val, label_temp_val = get_multlabel(metadata_path, val_id, labeldic)
    label_list_test, label_temp_test = get_multlabel(metadata_path, test_id, labeldic)
    #encode
    encoder = MultiLabelBinarizer()
    encoder.fit(label_list_train + label_list_val + label_list_test)
    train_label = encoder.transform(label_list_train)
    val_label = encoder.transform(label_list_val)
    test_label = encoder.transform(label_list_test)

    train_label = np.array(train_label)
    val_label = np.array(val_label)
    test_label = np.array(test_label)

    for id in range(len(train_id)):
        tt_path = os.path.join(label_path, str(train_id[id])+'.npy')
        np.save(tt_path, train_label[id])
    for id in range(len(val_id)):
        tt_path = os.path.join(label_path, str(val_id[id]) + '.npy')
        np.save(tt_path, val_label[id])
    for id in range(len(test_id)):
        tt_path = os.path.join(label_path, str(test_id[id]) + '.npy')
        np.save(tt_path, test_label[id])



