import json
import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn


class trailer_multimodal_features(Dataset):
    def __init__(self,
                 phase: str='train',
                 data_dir: str='',
                 feature_dir: str='',
                 visual_feature_dir: str='',
                 visual_feature_file: str=None,
                 cau_visual: bool = False,
                 audio_feature_dir: str=None,
                 audio_feature_file: str=None,
                 cau_audio: bool = False,
                 text_feature_dir: str=None,
                 text_token_file: str=None,
                 text_feature_file: str=None,
                 cau_text: bool = False,
                 meta_glo_file: str=None,
                 meta_loc_file: str=None,
                 num_classes: int=26,
                 ):
        self.phase = phase
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.visual_feature_dir = visual_feature_dir
        self.cau_visual = cau_visual
        self.audio_feature_dir = audio_feature_dir
        self.audio_feature_file = audio_feature_file
        self.cau_audio = cau_audio
        self.text_feature_dir = text_feature_dir
        self.text_token_file = text_token_file
        self.text_feature_file = text_feature_file
        self.cau_text = cau_text
        self.meta_glo_file = meta_glo_file
        self.meta_loc_file = meta_loc_file
        self.num_classes = num_classes

        # loading dataset samples
        with open(os.path.join(self.feature_dir, "data_{}_L.json".format(phase)), 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.samples = list(self.data.values())


        if self.visual_feature_dir is not None:
            if visual_feature_file is None:
                visual_feature_file="ViT-L-14-336_{}.pkl".format(phase)
            print('load visual features from:', os.path.join(self.visual_feature_dir, visual_feature_file))
            with open(os.path.join(self.visual_feature_dir, visual_feature_file) ,'rb') as handle:
                self.visual_feature_dict = pickle.load(handle)
            #aug_visual_feature_file="aug_ViT-L-14-336_{}.pkl".format(phase)
            aug_visual_feature_file = 'aug_' + visual_feature_file
            print('load aug visual features from:', os.path.join(self.visual_feature_dir, aug_visual_feature_file))
            with open(os.path.join(self.visual_feature_dir, aug_visual_feature_file) ,'rb') as handle:
                self.aug_visual_feature_dict = pickle.load(handle)
        if self.cau_visual == True:
            cau_visual_feature_file="cau_TG_{}.pkl".format(phase)
            print('load cau visual features from:', os.path.join(self.visual_feature_dir, cau_visual_feature_file))
            with open(os.path.join(self.visual_feature_dir, cau_visual_feature_file) ,'rb') as handle:
                self.cau_visual_feature_dict = pickle.load(handle)


        if self.audio_feature_file is not None:
            print('load audio features from:', self.audio_feature_file)
            with open(audio_feature_file, 'rb') as handle:
                self.audio_feature_dict = pickle.load(handle)

            aug_audio_feature_file= self.audio_feature_file.replace('ori', 'aug')
            print('load aug text features from:', aug_audio_feature_file)
            with open(aug_audio_feature_file,"rb") as handle:
                self.aug_audio_feature_dict = pickle.load(handle)
        if self.cau_audio == True:
            cau_audio_feature_file="cau_ori_audio_PANNs.pkl".format(phase)
            print('load cau text features from:', os.path.join(self.audio_feature_dir, cau_audio_feature_file))
            with open(os.path.join(self.audio_feature_dir, cau_audio_feature_file) ,'rb') as handle:
                self.cau_audio_feature_dict = pickle.load(handle)

            cau_aug_audio_feature_file="cau_aug_audio_PANNs.pkl".format(phase)
            print('load cau aug text features from:', os.path.join(self.audio_feature_dir, cau_aug_audio_feature_file))
            with open(os.path.join(self.audio_feature_dir, cau_aug_audio_feature_file) ,'rb') as handle:
                self.cau_aug_audio_feature_dict = pickle.load(handle)

        if self.text_feature_file is not None:
            print('load text features from:', self.text_feature_file)
            with open(text_feature_file,"rb") as handle:
                self.text_feature_dict = pickle.load(handle)

        if self.cau_text == True:
            cau_text_feature_file="cau_ori_text_Doc2Vec.pkl".format(phase)
            print('load cau text features from:', os.path.join(self.text_feature_dir, cau_text_feature_file))
            with open(os.path.join(self.text_feature_dir, cau_text_feature_file) ,'rb') as handle:
                self.cau_text_feature_dict = pickle.load(handle)

            cau_aug_text_feature_file="cau_aug_text_Doc2Vec.pkl".format(phase)
            print('load cau aug text features from:', os.path.join(self.text_feature_dir, cau_aug_text_feature_file))
            with open(os.path.join(self.text_feature_dir, cau_aug_text_feature_file) ,'rb') as handle:
                self.cau_aug_text_feature_dict = pickle.load(handle)
        if self.meta_glo_file is not None:
            print('load glo meta features from:', self.meta_glo_file)
            with open(meta_glo_file, 'rb') as handle:
                self.meta_glo_dict = pickle.load(handle)
        if self.meta_loc_file is not None:
            print('load loc meta features from:', self.meta_loc_file)
            with open(meta_loc_file, 'rb') as handle:
                self.meta_loc_dict = pickle.load(handle)
    def __getitem__(self, index: int):
        movie_id = self.keys[index]
        sample = self.samples[index]
        labels = np.array([int(ele) for ele in sample['label'].split(" ")])

        labels = torch.LongTensor(labels) # shape: (1)
        labels_onehot = nn.functional.one_hot(labels, num_classes=self.num_classes) # (postive_labels,num_classes)
        labels_onehot = labels_onehot.sum(dim=0).float() # (num_classes)

        return_dict = {
            "movie_id": movie_id,
            "label_onehot": labels_onehot,
        }

        if self.visual_feature_dir is not None:
            visual_feature_list = []
            aug_visual_feature_list = []
            for key, value in sample.items():
                if key == 'label': continue
                shot_frame0_path = value[0].replace(self.data_dir+"/","",1)
                shot_frame1_path = value[1].replace(self.data_dir+"/","",1)
                shot_frame2_path = value[2].replace(self.data_dir+"/","",1)

                shot_frame0_feature = self.visual_feature_dict[shot_frame0_path] # (1,512)
                shot_frame1_feature = self.visual_feature_dict[shot_frame1_path] # (1,512)
                shot_frame2_feature = self.visual_feature_dict[shot_frame2_path] # (1,512)

                visual_feature_list.append(np.mean([shot_frame0_feature, shot_frame1_feature, shot_frame2_feature], axis=0)) # (1,512)

                shot_frame0_feature = self.aug_visual_feature_dict[shot_frame0_path] # (1,512)
                shot_frame1_feature = self.aug_visual_feature_dict[shot_frame1_path] # (1,512)
                shot_frame2_feature = self.aug_visual_feature_dict[shot_frame2_path] # (1,512)
                aug_visual_feature_list.append(np.mean([shot_frame0_feature, shot_frame1_feature, shot_frame2_feature], axis=0))

            #has only 1-2 shot
            if len(visual_feature_list) < 8:
                num = 8-len(visual_feature_list)
                for _ in range(num):
                    visual_feature_list.append(visual_feature_list[0])
            visual_features = np.concatenate(visual_feature_list ,axis=0) # (5,512)-> (num_of_shot, feat_embed_size)
            return_dict["visual_feature"] = visual_features
            if len(aug_visual_feature_list) < 8:
                num = 8-len(aug_visual_feature_list)
                for _ in range(num):
                    aug_visual_feature_list.append(aug_visual_feature_list[0])
            aug_visual_features = np.concatenate(aug_visual_feature_list , axis=0)
            return_dict["aug_visual_feature"] = aug_visual_features
        if self.cau_visual == True:
            cau_visual_feature_list = []
            #cau_aug_visual_feature_list = []
            for key, value in sample.items():
                if key=='label': continue
                shot_frame0_path = value[0].replace(self.data_dir+"/","",1)
                shot_frame0_path = shot_frame0_path.split('/')[-2]
                shot_frame0_feature = self.cau_visual_feature_dict[shot_frame0_path]
                cau_visual_feature_list.append(shot_frame0_feature)
                # shot_frame0_feature = self.cau_aug_visual_feature_dict[shot_frame0_path]  # (1,512)
                # cau_aug_visual_feature_list.append(shot_frame0_feature)
            if len(cau_visual_feature_list) < 8:
                num = 8-len(cau_visual_feature_list)
                for _ in range(num):
                    cau_visual_feature_list.append(cau_visual_feature_list[0])
            cau_visual_features = np.concatenate(cau_visual_feature_list , axis=0)
            return_dict["cau_visual_feature"] = cau_visual_features

        if self.text_feature_file is not None:
            text_features = self.text_feature_dict[movie_id]


            return_dict["topic_feature"] = text_features

        if self.cau_text:
            cau_text_features = self.cau_text_feature_dict[movie_id]
            return_dict["cau_text_feature"] = cau_text_features

            cau_aug_text_features = self.cau_aug_text_feature_dict[movie_id]
            return_dict["cau_aug_text_feature"] = cau_aug_text_features
        if self.audio_feature_file is not None:
            audio_features = self.audio_feature_dict[movie_id]
            return_dict["audio_feature"] = audio_features # (2048,), PANNs feature

            aug_audio_features = self.aug_audio_feature_dict[movie_id]
            return_dict["aug_audio_feature"] = aug_audio_features
        if self.cau_audio:
            cau_audio_features = self.cau_audio_feature_dict[movie_id]
            return_dict["cau_audio_feature"] = cau_audio_features

            cau_aug_audio_features = self.cau_aug_audio_feature_dict[movie_id]
            return_dict["cau_aug_audio_feature"] = cau_aug_audio_features
        if self.meta_glo_file is not None:
            meta_glo = self.meta_glo_dict[movie_id]
            return_dict["meta_glo"] = meta_glo
        if self.meta_loc_file is not None:
            meta_loc = self.meta_loc_dict[movie_id]
            return_dict["meta_loc"] = meta_loc
        return return_dict

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    sf = 12
    # path = '../../data/MovieNet/aug\\tt0035423\\shot_0229_img_0.jpg'
