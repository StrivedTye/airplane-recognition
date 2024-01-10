import argparse
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import model.net as MODEL
from test.SegmentScene import SegmentScene
from test.TestPartSegmentation import seg_support
from util import provider
import yaml
import warnings
import glob
import open3d as o3d
import os
import copy
from test.PostProcess import post_process_ab

warnings.filterwarnings('ignore')


class TestClassification:
    def __init__(self, model, device):
        self.class_reference = ['A320-200', 'A330-200', 'A340-200', 'A380-800', 'B737-200',
                                'B737-400', 'B747-200', 'B757-200', 'B777-200', 'B737-800']
        self.distance_reference = [11.84, 20.84, 25.87, 32.58, 14.52,
                                   10.72, 31.06, 17.59, 21.08, 15.73]
        self.model = model
        self.device = device
        self.votes = []

    def reset(self):
        self.votes = []

    def predict_type(self, points):
        pcd = points.copy()
        points = np.expand_dims(points, axis=0)
        points = provider.normalize_data(points)
        points = torch.Tensor(points).transpose(2, 1)
        points = points.to(self.device)
        with torch.no_grad():
            pred, _, seg, _, scale, ab = self.model(points)  # transpose for pointnet, not DGCNN

        pred_from_seg, _ = seg_support(pcd=pcd, seg=seg, w=1.0, distance_reference=self.distance_reference)
        pred = post_process_ab(pred=pred, ab=ab)

        pred = torch.softmax(pred, dim=1)
        # mask = (pred_from_seg.to(self.device) > 0.1).double()
        pred_ID = torch.argmax(pred.squeeze().double()).detach().item()  # pred.squeeze().double() + pred_from_seg.to(self.device)
        pred_type = self.class_reference[pred_ID]
        ab = ['B', 'A'][torch.argmax(torch.softmax(ab, dim=1)).detach().item()]
        return pred_ID, pred_type, ab

    def voting(self, gt_type="A320-200"):
        decision = self.publicnum(self.votes)
        # decision = self.class_reference[decision]
        return decision, decision == gt_type

    def update(self, pred_ID):
        self.votes.append(pred_ID)

    @staticmethod
    def publicnum(num):
        dictnum = {}
        for i in range(len(num)):
            if num[i] in dictnum.keys():
                dictnum[num[i]] += 1
            else:
                dictnum.setdefault(num[i], 1)
        maxnum = 0
        maxkey = 0
        for k, v in dictnum.items():
            if v > maxnum:
                maxnum = v
                maxkey = k
        return maxkey


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('TestClassification')
    parser.add_argument("--config", type=str, default="../config/test.yaml", help="config file")
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--model', default='pointnet', help='model name [default: pointnet_cls]')
    parser.add_argument('--data_dir', type=str,
                        default='/media/mark/Elements SE/test_plane/5-B737-800-snow/20201228_090134_DEFAULT',
                        help='dataset path')  # 20201108_131556_DEFAULT-a320-200,  # 20201022_083935_DEFAULT-b737-800
    parser.add_argument('--type', type=str, default='real', choices=['real', 'fake'],
                        help='Date Type, if real, z and y axis will be exchanged')
    parser.add_argument('--bridge', action='store_false', help='Set if there has no bridge in the scene')
    parser.add_argument('--checkpoints_path', type=str, default='../train/log/classification/2020-12-23_15-28/checkpoints/best_model.pth',  # 2020-12-08_09-39
                        help='Pretrained Model Path')
    parser.add_argument('--ground_high', type=float, default=-4, help='the ground truth higth')
    parser.add_argument('--angle', type=float, nargs='+', default=[-0.5, 0, -2.5])
    parser.add_argument('--begin', type=int, default=0,
                        help='the first frame to begin classification and segmentation')
    parser.add_argument('--end', type=int, default=0,
                        help='the end frame to end classification and segmentation, if set -1, it will end at the last frame')
    return parser.parse_args()


def main():
    # Define Model
    args = parse_args()
    print("\n**************************")
    for k, v in args._get_kwargs():
        print('[%s]:' % (k), v)
    print("**************************\n")

    pretrained_path = args.checkpoints_path  # 10-30-09-56
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    net = MODEL.get_model(model=args.model, normal_channel=False).to(device)
    checkpoints = torch.load(pretrained_path)
    net.load_state_dict(checkpoints['model_state_dict'])
    net.eval()

    # init the class
    gt_type = 'B737-800'
    recognition = TestClassification(model=net, device=device)

    # new incoming data
    scene_seg = SegmentScene(config_file=args.config)
    for idx, file in tqdm(enumerate(sorted(glob.glob(os.path.join(args.data_dir, '*.pcd'))))):
        if idx < args.begin or (not args.end == 0 and idx > args.end):
            continue
        if idx == args.begin:
            scene = np.array(o3d.io.read_point_cloud(file).points)
            scene_seg.reset(scene)
        scene = np.array(o3d.io.read_point_cloud(file).points)
        scene, _, _, plane, _ = scene_seg.forward(scene)
        # update the instance
        if plane is None:
            continue
        cur_pred_ID, cur_pred_type, ab = recognition.predict_type(plane)
        print(cur_pred_type, ab)
        pred_type = recognition.update(cur_pred_ID)
        recognition.update(pred_type)
        # recognition
        predicted_type, result = recognition.voting(gt_type)  # result is a bool val
    print('The predicted airplane type is {}'.format(predicted_type))
    print('The predicted result is {}'.format(result))


if __name__ == '__main__':
    main()
