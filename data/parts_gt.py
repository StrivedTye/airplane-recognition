import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def parts_gt(part_labels):
    part_labels = part_labels.squeeze()
    head = ((part_labels - torch.ones(128, 256)*1702063982) == 0)
    body_small, body_big = (part_labels-torch.ones(128, 256)*1886546273) == 0, (part_labels-torch.ones(128, 256)*2036625250) == 0
    body = body_small + body_big
    engine_small1, engine_small2, engine_big1, engine_big3, engine_big2, engine_big4, engine7 = \
        (part_labels-torch.ones(128, 256)*1850040146) == 0, (part_labels-torch.ones(128, 256)*1852137292) == 0, \
        (part_labels-torch.ones(128, 256)*1970233170) == 0, (part_labels-torch.ones(128, 256)*1970233164) == 0, \
        (part_labels-torch.ones(128, 256)*1852399442) == 0, (part_labels-torch.ones(128, 256)*1852399436) == 0, \
        (part_labels-torch.ones(128, 256)*1852137298) == 0
    engine = engine_small1 + engine_small2 + engine_big1 + engine_big3 + engine_big2 + engine_big4 + engine7
    wing1, wing2, wing3 = (part_labels-torch.ones(128, 256)*1767333708) == 0, (part_labels-torch.ones(128, 256)*1769430866) == 0, (part_labels-torch.ones(128, 256)*1769430860) == 0
    wing = wing1 + wing2 + wing3
    tail = (part_labels-torch.ones(128, 256)*1818845556) == 0
    outlier = (part_labels-torch.ones(128, 256)*1) == 0
    head_gt, body_gt, engine_gt, wing_gt, tail_gt, outlier_gt = head.float().cuda(), body.float().cuda(), engine.float().cuda(), \
        wing.float().cuda(), tail.float().cuda(), outlier.float().cuda()
    seg_gt = head_gt*0 + body_gt*1 + engine_gt*2 + wing_gt*3 + tail_gt*4 + outlier_gt*5
    return seg_gt.reshape(128*256).long()


def scale_loss(target, pred_distance):
    distance_reference = torch.tensor([11.84, 10.72, 20.84, 15.73, 25.87, 31.06, 17.59, 32.58, 14.52, 21.80]).cuda()
    distances = distance_reference[target]
    pred_distance = pred_distance.squeeze(dim=-1)
    return torch.sum(torch.abs((distances - pred_distance)))/128


# def scale_loss(seg_gt, points, pred):
#     # seg_gt: (128, 256), 0,1,...,5; points: (128, 256, 3); pred: (128, 10)
#     # compute distances
#     head_, engine_ = (seg_gt == 0).float().cuda(), (seg_gt == 2).float().cuda()  # (128, 256)
#     try:
#         head_weight, engine_weight = 1.0/torch.sum(head_, dim=1), 1.0/torch.sum(engine_, dim=1)  # (128)
#     except:
#         return 0
#     head_center = head_weight.unsqueeze(dim=-1).repeat(1, 3) * torch.sum(points * (head_.unsqueeze(dim=-1).repeat(1, 1, 3)), dim=1)  # (128, 3)
#     engine = points * (engine_.unsqueeze(dim=-1).repeat(1, 1, 3))  # (128, 256, 3)
#     res = engine - head_center.unsqueeze(dim=1).repeat(1, 256, 1)  # (128, 256, 3)
#     res = res * (engine_.unsqueeze(dim=-1).repeat(1, 1, 3))  # (128, 256, 3)
#     res = torch.norm(res, dim=-1)  # (128, 256)
#     distances = engine_weight * torch.sum(res, dim=1)
#
#     # pred distances
#     pred = F.log_softmax(pred, dim=-1)
#     pred_one_hot = F.gumbel_softmax(logits=pred, tau=0.01, hard=True, eps=1e-10)
#     distance_reference = torch.tensor([[11.84, 10.72, 20.84, 15.73, 25.87, 31.06, 17.59, 32.58, 14.52, 21.80]]).repeat(128, 1).cuda()
#     pred_distances = torch.sum(pred_one_hot * distance_reference, dim=-1)
#     non_nan = torch.isnan(distances) == 0
#     return torch.sum(torch.abs((distances - pred_distances)[non_nan]))/torch.sum(non_nan)


# def parts_gt(part_labels):
#     head = (part_labels - 1702063982) == 0
#     body_small, body_big = part_labels-1886546273 == 0, part_labels-2036625250 == 0
#     body = body_small + body_big
#     engine_small1, engine_small2, engine_big1, engine_big3, engine_big2, engine_big4, engine7 = \
#         (part_labels-1850040146) == 0, (part_labels-1852137292) == 0, \
#         (part_labels-1970233170) == 0, (part_labels-1970233164) == 0, \
#         (part_labels-1852399442) == 0, (part_labels-1852399436) == 0, \
#         (part_labels-1852137298) == 0
#     engine = engine_small1 + engine_small2 + engine_big1 + engine_big3 + engine_big2 + engine_big4 + engine7
#     wing1, wing2, wing3 = (part_labels-1767333708) == 0, (part_labels-1769430866) == 0, (part_labels-1769430860) == 0
#     wing = wing1 + wing2 + wing3
#     tail = (part_labels-1818845556) == 0
#     outlier = (part_labels-1) == 0
#     seg_gt = head*0 + body*1 + engine*2 + wing*3 + tail*4 + outlier*5
#     return seg_gt
