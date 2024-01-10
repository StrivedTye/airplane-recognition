# written by Tye 09/22/2020.


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

# from model.pointnet_plus import PointNet2
from model.pointnet import PointNetClsSeg, feature_transform_regularizer
from model.dgcnn import DGCNNClsSeg


class get_model(nn.Module):

    def __init__(self, model="pointnet", num_cls=10, normal_channel=True):
        super(get_model, self).__init__()

        if normal_channel:
            channel = 6
        else:
            channel = 3

        # Backbone
        if model == "DGCNN":
            self.model = DGCNNClsSeg(n_emb_dims=512, num_cls=num_cls, p=0.5, num_neighbor=10)
        else:
            self.model = PointNetClsSeg(channel=channel, feature_transform=True, num_cls=num_cls, p=0.5)

    def forward(self, points):
        '''

        :param points: [128, 3, 256]
        :return:
        '''

        cls, trans_feat, seg, main_axis, scale, ab = self.model(points)

        return cls, trans_feat, seg, main_axis, scale, ab


class get_loss(nn.Module):

    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat=None):

        mat_diff_loss = feature_transform_regularizer(trans_feat) if trans_feat is not None else 0

        # loss = self.cal_loss(pred, target, smoothing=True)
        loss = F.nll_loss(input=F.log_softmax(pred, dim=1), target=target)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

    def cal_loss(self, pred, target, smoothing=True):
        """
        Calculate cross entropy loss, apply label smoothing if needed.
        :param pred: predicted tensor, of shape [B, num_cls]
        :param target: ground-truth, of shape [B]
        :param smoothing:
        :return:
        """

        target = target.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, target, reduction='mean')

        return loss

