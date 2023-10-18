import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionLayer(nn.Module):
    def __init__(self, hid_c,num_vertices):
        super(SpatialAttentionLayer, self).__init__()
        self.W_1 = nn.Parameter(torch.randn(size=(hid_c, hid_c)))
        self.W_2 = nn.Parameter(torch.randn(size=(hid_c, hid_c)))
        self.b_s = nn.Parameter(torch.randn(size=(1, num_vertices, num_vertices)))
        self.V_s = nn.Parameter(torch.randn(size=(num_vertices, num_vertices)))

    def forward(self, x):  # X [B,N,hid_c]
        lhs = torch.matmul(x, self.W_1)
        # shape of rhs is (batch_size,N,N)
        rhs = torch.matmul(self.W_2, x.transpose(1,2))  # [B,6,N]
        product = torch.matmul(lhs, rhs) + self.b_s
        S = torch.matmul(self.V_s, torch.sigmoid(product))
        # normal
        sum_s = torch.sum(S, dim=0)
        S_normal = torch.exp(S) / torch.exp(torch.sum(sum_s))

        return S_normal


