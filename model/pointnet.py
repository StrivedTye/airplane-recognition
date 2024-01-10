import torch
import torch.nn as nn
import torch.nn.functional as F

from model.stn import STN3d, STNkd


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


class PointNetEncoder(nn.Module):
    def __init__(self, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()

        self.stn = STN3d(channel)

        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.feature_transform = feature_transform

        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3, dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))
        pw_feat = x  # [128, 1024, 256]

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x_repeat = x.view(-1, 1024, 1).repeat(1, 1, N)
        return x, trans, trans_feat, torch.cat([x_repeat, pw_feat], 1)


class PointNetClsSeg(nn.Module):
    def __init__(self, channel=3, feature_transform=False, num_cls=10, p=0.5):
        super(PointNetClsSeg, self).__init__()

        # extract feature
        self.feat = PointNetEncoder(feature_transform=feature_transform, channel=channel)

        # classification
        self.cls = nn.Sequential(
                                 nn.Linear(1024, 512),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=p),
                                 nn.Linear(512, 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256, num_cls),
                                 )

        # segmentation
        self.seg = nn.Sequential(nn.Conv1d(2048, 512, 1),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=p),
                                 nn.Conv1d(512, 256, 1),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(256, 6, 1))

        self.main_axis = nn.Sequential(nn.Linear(1024, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, 256),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(256, 3))

        # self.scale = nn.Sequential(nn.Linear(1024, 512),
        #                            nn.BatchNorm1d(512),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(512, 256),
        #                            nn.BatchNorm1d(256),
        #                            nn.ReLU(inplace=True),
        #                            nn.Linear(256, 1),
        #                            )

        self.ab = nn.Sequential(nn.Linear(1024, 512),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(512, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(256, 2),
                                   )

    def forward(self, points):

        global_feat, _, trans_feat, pw_feat = self.feat(points)
        cls = self.cls(global_feat)
        seg = self.seg(pw_feat)
        main_axis = self.main_axis(global_feat)
        scale = None
        ab = self.ab(global_feat)
        return cls, trans_feat, seg, main_axis, scale, ab

