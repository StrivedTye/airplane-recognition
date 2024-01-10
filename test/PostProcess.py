import torch


def post_process_ab(pred, ab):
    ab = ab.squeeze()
    pred = torch.softmax(pred, dim=1)
    # type A
    if ab[0] < ab[1]:
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).cuda()
    # type B
    else:  #
        mask = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).cuda()
    return pred * mask


if __name__ == '__main__':
    pred = torch.randn(10).unsqueeze(dim=0).cuda()
    ab = torch.randn(2).unsqueeze(dim=0).cuda()
    print('pred: ', pred, '\nab: ', ab, '\nprocessed: ', post_process_ab(pred=pred, ab=ab))
