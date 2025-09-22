import torch
import torch.nn as nn
import torchvision.models as models
import math
from model.RA_T import RA_T
def DensewithBN(in_fea, out_fea, normalize=True, dropout=False):
    layers = [nn.Linear(in_fea, out_fea)]
    if normalize == True:
        layers.append(nn.BatchNorm1d(num_features=out_fea))
    layers.append(nn.ReLU())
    #
    if dropout:
        layers.append(nn.Dropout(p=0.5))
    return layers
def ZFFN(in_fea, out_fea, normalize=True, dropout=False):
    layers = [nn.Linear(in_fea, out_fea)]
    if normalize == True:
        layers.append(nn.BatchNorm1d(num_features=out_fea))
    layers.append(nn.ReLU())
    if dropout:
        layers.append(nn.Dropout(p=0.5))
    return layers

class JEM(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, meta_num, ori_len, a_len, cau_len, type_num, num_category):
        super(JEM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.meta_num = meta_num
        self.ori_len = ori_len
        self.a_len = a_len
        self.cau_len = cau_len
        self.type_num = type_num
        self.num_category = num_category
        self.z_len = self.cau_len
        #
        self.transformer_encoder = RA_T(num_layers=self.num_layers, d_model=self.hidden_dim,
                                                          n_heads=self.num_heads, dim_feedforward=768, dropout=0.1)

        #type embedding
        self.type_embedding = torch.nn.Embedding(self.type_num, self.hidden_dim)

        self.z_dense = nn.Sequential(*DensewithBN(self.z_len * 2, self.z_len * 2, dropout=True))  #
        self.z_ffn = nn.Sequential(*ZFFN(self.z_len * 2, self.z_len * 2, dropout=True))  #
        self.out_linear = nn.Linear(self.z_len * 2, self.num_category)
        self.a_linear = nn.Linear(self.a_len, self.ori_len)


        self.linear1 = nn.Linear(self.z_len * 2, self.z_len * 2)
        self.linear2 = nn.Linear(self.z_len * 2, self.z_len * 2)

        self.dropout = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = nn.ReLU()  #


        self.norm1 = nn.LayerNorm(self.z_len * 2)
        self.norm2 = nn.LayerNorm(self.z_len * 2)


        self.meta_embedding = nn.Linear(self.meta_num, self.ori_len)#768
    def forward(self, f_one,type_index, pe, lambda_f, lambda_m):
        mean_visual_features = torch.mean(f_one['visual_feature'], dim=1)  # b * 768
        mean_VTM_visual_features = torch.mean(f_one['VTM_visual_feature'], dim=1)  # b *
        mean_cau_visual_features = torch.mean(f_one['cau_visual_feature'], dim=1)  # b *

        text_feature = f_one['topic_feature'].squeeze(1)
        VTM_text_feature = f_one['VTM_topic_feature'].squeeze(1)


        a_features = mean_VTM_visual_features + 0.2 * text_feature
        b_features = mean_cau_visual_features

        z_features = torch.cat((a_features, b_features), dim=1)
        z_features = self.z_dense(z_features)
        z_features = z_features.unsqueeze(-1)
        # z_features_d = z_features_d.unsqueeze(-1)

        # m_output = self.embedding(m_output)
        z_features = self.transformer_encoder(z_features)
        type_encoding = self.type_embedding(type_index)
        type_encoding = type_encoding.unsqueeze(0).expand(z_features.shape[0], self.z_len * 2, -1)
        z_features = z_features + type_encoding + pe
        z_features = z_features.mean(-1)

        output = self.out_linear(z_features)
        return output

