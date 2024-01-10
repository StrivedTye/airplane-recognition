import torch
import numpy as np
import sys
sys.path.append('/home/mark/NAS/airplane-code-master')
from util import provider
from util.visualization import mlabvis
import model.net as MODEL


def segment(points, model, device='cuda:0'):
    r"""
    :param points: np.array (num_points, 3)
    :param model: a .eval() mode model
    :param device: cuda name
    :return: a dict record the segmentation result
    """
    # prepare input
    pcd = points.copy()
    points = np.expand_dims(points, axis=0)  # (1, num_points, 3)
    points = provider.normalize_data(points)
    points = torch.Tensor(points).transpose(2, 1)  # (1, 3, num_points)
    points = points.to(device)
    result = {}

    # compute segmentation
    _, _, seg, _, _ = model(points)  # seg: (1, 6, num_points)
    seg = seg[0].transpose(1, 0)
    seg = torch.argmax(seg, dim=1).cpu().numpy()  # (num_points) 0, 1, 2, 3, 4, 5

    result['head'] = pcd[seg == 0]
    result['body'] = pcd[seg == 1]
    result['engine'] = pcd[seg == 2]
    result['wing'] = pcd[seg == 3]
    result['tail'] = pcd[seg == 4]
    result['outlier'] = pcd[seg == 5]

    return result


def seg_support(pcd, seg, w=0.5, distance_reference=None):
    distance_reference = np.array(distance_reference)

    # compute mean distance from seg result
    seg = seg[0].transpose(1, 0)
    seg = torch.argmax(seg, dim=1).cpu().numpy()  # (num_points) 0, 1, 2, 3, 4, 5
    head, engine = pcd[seg == 0], pcd[seg == 2]  # [N, 3]
    if not head.shape[0] * engine.shape[0]:
        return torch.tensor(distance_reference * 0), -1
    head_center = np.expand_dims(np.mean(head, axis=0), axis=0).repeat(engine.shape[0], axis=0)
    distances = np.linalg.norm(engine - head_center, axis=1)
    mean_distance = np.mean(distances)

    # compute confidence from mean distance
    res = np.abs(distance_reference - mean_distance)
    score = 1/(res+0.0001)
    score = score/np.sum(score)
    return torch.tensor(w * score), mean_distance


def visualize_segmentation(result):
    head, body, engine, wing, tail, outliers = result['head'], result['body'], result['engine'], \
                                               result['wing'], result['tail'], result['outlier']
    mlabvis(pcd=head, pcd2=body, pcd3=engine, pcd4=wing, pcd5=tail, pcd6=outliers)


if __name__ == '__main__':
    # load trained model for testing
    pretrained_path = '/home/mark/NAS/airplane-code-master/train/log/classification/2020-12-06_16-12/checkpoints/best_model.pth'
    net = MODEL.get_model(normal_channel=False, model='DGCNN').cuda()
    checkpoints = torch.load(pretrained_path)
    net.load_state_dict(checkpoints['model_state_dict'])
    net.eval()
    torch.cuda.set_device(0)

    # # loaded data
    path = '/media/mark/dc542bc0-6d21-45c0-b1e1-2c7bacfc68ed/planeSet_crop/data1/A340-200/A340-2009gaussian/scan00200.bin'
    points = np.fromfile(path, dtype=np.float32)
    points = points.reshape(-1, 4)[:, :3]
    # print(points)
    # result = segment(points=points, model=net)
    #
    # # visualize result
    # visualize_segmentation(result=result)

    result = segment(points=points, model=net)
    seg_support(points=points, seg=seg)
