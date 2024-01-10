# Written by Tye, 09/23/2020

from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.stn import TransformNet


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=10):
    # x = x.squeeze()
    x = x.view(*x.size()[:3])
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


class DGCNN(nn.Module):
    def __init__(self, n_emb_dims=512, k=10):
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2 , 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(n_emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        xyz = x.unsqueeze(dim=-1).repeat(1, 1, 1, 10)  # (B, 3, num_points, 10)

        x = get_graph_feature(x, k=self.k)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x1, k=self.k)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x2, k=self.k)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x3, k=self.k)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # xyz added for scale info

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2).view(batch_size, -1, num_points)

        # for point-wise feature
        x_ = x.max(dim=-1, keepdim=True)[0]  # (B, emd_dim, 1)
        x_ = x_.repeat(1, 1, num_points).unsqueeze(dim=-1)
        pw_feat = torch.cat((x_, x1, x2, x3, x4), dim=1)

        # for global feature
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        global_feat = torch.cat((x1, x2), dim=1)
        return global_feat, pw_feat


class DGCNNClsSeg(nn.Module):
    def __init__(self, n_emb_dims, num_cls=10, p=0.5, num_neighbor=10):
        super(DGCNNClsSeg, self).__init__()
        self.k = num_neighbor

        self.stn = TransformNet()

        self.feat = DGCNN(n_emb_dims=512, k=num_neighbor)

        self.cls = nn.Sequential(nn.Linear(n_emb_dims * 2, 512, bias=False),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Dropout(p=p),
                                 nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Dropout(p=p),
                                 nn.Linear(256, num_cls))

        # segmentation
        # self.xyz_trans = nn.Sequential(nn.Conv1d(3, 64, 1),
        #                          nn.BatchNorm1d(64),
        #                          nn.LeakyReLU(inplace=True),
        #                          nn.Dropout(p=p),
        #                          nn.Conv1d(64, 128, 1),
        #                          nn.BatchNorm1d(128),
        #                          nn.LeakyReLU(inplace=True),
        #                          nn.Conv1d(128, 256, 1))

        # segmentation
        self.seg = nn.Sequential(nn.Conv1d(1024, 512, 1),
                                 nn.BatchNorm1d(512),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Dropout(p=p),
                                 nn.Conv1d(512, 256, 1),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Conv1d(256, 6, 1))

        self.main_axis = nn.Sequential(nn.Linear(1024, 512),
                                       nn.BatchNorm1d(512),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(512, 256),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(256, 3))

    def forward(self, x):
        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3, num_points)
        t = self.stn(x0)                        # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        # scale
        # xyz_trans = F.max_pool1d(self.xyz_trans(x), x.shape[-1])  # (batch_size, 256, num_points)
        # xyz_trans = xyz_trans.reshape(xyz_trans.shape[0], xyz_trans.shape[1])

        global_feat, pw_feat = self.feat(x)     # (batch_size,  1024)
        cls = self.cls(global_feat)            # (batch_size, num_class)
        seg = self.seg(pw_feat.reshape(pw_feat.shape[0:3]))       # (batch_size, num_seg, num_points)
        main_axis = self.main_axis(global_feat) # (batch_size, 3)
        return cls, None, seg, main_axis
