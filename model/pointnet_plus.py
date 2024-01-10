# Written by Tye, 23/09/2020

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

from model.pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetFPModule


# PointNet++
class PointNet2(nn.Module):
    """get point-wise feature"""

    def __init__(self, input_channels=3, use_xyz=True):
        super(PointNet2, self).__init__()

        skip_channel_list = [input_channels, 128, 256, 256]

        self.SA_module = nn.ModuleList()
        self.SA_module.append(
            PointnetSAModule(mlp=[input_channels, 64, 64, 128],
                             radius=0.3,
                             nsample=32,
                             bn=True,
                             use_xyz=use_xyz)
        )
        self.SA_module.append(
            PointnetSAModule(mlp=[128, 128, 128, 256],
                             radius=0.5,
                             nsample=32,
                             bn=True,
                             use_xyz=use_xyz)
        )
        self.SA_module.append(
            PointnetSAModule(mlp=[256, 256, 256, 256],
                             radius=0.7,
                             nsample=32,
                             bn=True,
                             use_xyz=use_xyz)
        )

        self.trans = nn.Conv1d(256, 256, kernel_size=1)

        self.FP_module = nn.ModuleList()
        self.FP_module.append(
            PointnetFPModule(mlp=[256 + skip_channel_list[-2], 128, 128], bn=True)
        )
        self.FP_module.append(
            PointnetFPModule(mlp=[128 + skip_channel_list[-3], 256, 256], bn=True)
        )
        self.FP_module.append(
            PointnetFPModule(mlp=[256 + skip_channel_list[-4], 256, 256], bn=True)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_module)):
            li_xyz, li_features = self.SA_module[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        mid_feat = self.trans(l_features[-1])
        mid_xyz = l_xyz[-1]
        for i in range(len(self.FP_module)):
            j = -(i + 1)
            l_features[j - 1] = self.FP_module[i](l_xyz[j - 1], l_xyz[j], l_features[j - 1], l_features[j])

        return l_xyz[0], l_features[0], mid_xyz, mid_feat  # [B, N, 3], [B, C, N]
