import torch
import torch.nn as nn
class FRB(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # 前馈网络（FFN）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):

        src_roll = torch.roll(src, shifts=1, dims=1)
        src_d = (src + src_roll) / 2.0

        src_d2 = self.norm1(src_d)
        src_d2, _ = self.self_attn(src_d2, src_d2, src_d2, attn_mask=src_mask)
        src_d = src_d + self.dropout1(src_d2)

        src_d2 = self.norm2(src_d)
        src_d2 = self.linear2(self.dropout(self.activation(self.linear1(src_d2))))
        src_d = src_d + self.dropout2(src_d2)

        return src_d



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # 前馈网络（FFN）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.linear4 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class RA_T(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.layers1 = TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
        self.layers2 = FRB(d_model, n_heads, dim_feedforward, dropout)

    def forward(self, src, src_mask=None):
        src1 = self.layers1(src, src_mask)
        src2 = self.layers2(src1, src_mask)
        out = 0.5*src1 + 0.5*src2
        return out
