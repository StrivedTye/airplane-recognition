# Written by Tye, 09/22/2020
# Modified by Tye, 12/01/2020

import os.path as osp
import numpy as np
import glob
import random
import torch
from torch.utils.data import Dataset
# from util.visualization import vis_with_boxes as vis
from util.point_cloud import PointsProcess
from data.augmentor import augmentor
from data.parts_gt import parts_gt
from data.get_labels import soften


PLANE_LIST = ['A320-200', 'A330-200', 'A340-200', 'A380-800', 'B737-200',
              'B737-400', 'B747-200', 'B757-200', 'B777-200', 'B737-800']
DATA_LIST = ['data1', 'data2', 'data3', 'data4']
NOISE_LIST = ['0gaussian', '0laplace', '1gaussian', '1laplace',
              '2gaussian', '2laplace', '3gaussian', '3laplace',
              '4gaussian', '4laplace', '5gaussian', '5laplace',
              '6gaussian', '6laplace','7gaussian', '7laplace',
              '8gaussian', '8laplace','9gaussian', '9laplace']
PLANE_DICT = {plane: i for i, plane in enumerate(PLANE_LIST)}
DATA_DICT = {data: i for i, data in enumerate(DATA_LIST)}
NOISE_DICT = {noise: i for i, noise in enumerate(NOISE_LIST)}


class PlaneDataset(Dataset):
    def __init__(self, root_path, split='train'):
        self.train = split

        self.root_path = root_path

        self.model_dict = PLANE_DICT
        # self.model_dict = {'A320-200': 0, 'B737-400': 5, 'A330-200': 1, 'B737-800': 5, 'A340-200': 1,
        #                    'B747-200': 3, 'B757-200': 4, 'A380-800': 2, 'B737-200': 5, 'B777-200': 3}

        if split == 'train':
            ids = self._get_split_data('train')
        elif split == 'test':
            ids = self._get_split_data('test')
        else:  # validation
            ids = self._get_split_data('val')

        # Load pos file
        self.all_angle = np.load('../data/pos.npy')

        # Get path list which contains all plane instances
        self.path_list = self._load_path_list(ids)

        random.shuffle(self.path_list)

    def __len__(self):
        return len(self.path_list)

    def _get_split_data(self, split):
        if split == 'train':
            return range(0, 8)
        elif split == 'test':
            return range(8, 10)
        else:
            return range(0,2)

    def _get_type(self, path=''):
        """ get model type represented by '0-9' """

        type = 0
        for k, v in self.model_dict.items():
            if path.find(k) != -1:
                type = v
                break
        return type

    def _get_pos(self, path=''):
        path_split = path.split('/')
        type_name = path_split[-3]

        # a = random.choice(['data1', 'data2', 'data3', 'data4'])
        a = random.choice(['data1', 'data2', 'data3'])
        b = random.choice(['gaussian', 'laplace'])
        c = random.choice(range(10))
        d = random.choice(range(1, 241))
        d = 'scan' + '{:0>5d}'.format(d) + '.bin'

        path_split[-1] = d
        path_split[-2] = type_name + str(c) + b
        path_split[-4] = a

        final_path = '/'
        for p in path_split:
            final_path = final_path + '/' + p

        return osp.join(final_path)

    def _get_neg(self, path=''):
        path_split = path.split('/')
        type_name = path_split[-3]
        diff_set = [k for k, v in self.model_dict.items() if k != type_name]
        new_type_name = random.choice(diff_set)

        a = random.choice(['data1', 'data2', 'data3'])
        b = random.choice(['gaussian', 'laplace'])
        c = random.choice(range(10))
        d = random.choice(range(1, 241))
        d = 'scan' + '{:0>5d}'.format(d) + '.bin'

        path_split[-1] = d
        path_split[-2] = new_type_name + str(c) + b
        path_split[-3] = new_type_name
        path_split[-4] = a

        final_path = '/'
        for p in path_split:
            final_path = final_path + '/' + p

        return osp.join(final_path)

    def _get_yaw(self, path=''):
        a, b, c, d = path.split('/')[-4:]
        data = DATA_DICT[a]
        plane = PLANE_DICT[b]
        noise = NOISE_DICT[c[8:]]
        id = int(d.split('.')[0][-4:]) - 1
        yaw = self.all_angle[data][plane][noise][id]
        return yaw

    def _load_path_list(self, ids=range(1)):
        path_list = []
        scenes = glob.glob(osp.join(self.root_path, 'data1')) + glob.glob(osp.join(self.root_path, 'data2'))
        for scene in sorted(scenes):  # different scenes: sunny or rainy; bridge or not
            for plane_type in glob.glob(osp.join(scene, '*')):  # plane type: A340-200
                plane_type_name = plane_type.split('/')[-1]
                for i in ids:  # 0-9
                    # get the path of each frame
                    gau_path = osp.join(plane_type, plane_type_name + str(i) + 'gaussian', '*.bin')
                    lap_path = osp.join(plane_type, plane_type_name + str(i) + 'laplace', '*.bin')
                    gau_pcds = glob.glob(gau_path)  # produced by gaussian noisy
                    lap_pcds = glob.glob(lap_path)  # produced by laplace noisy
                    # gau_pcds = list(filter(lambda x: 170 < int(x[-9:-4]) < 181, gau_pcds))
                    # lap_pcds = list(filter(lambda x: 170 < int(x[-9:-4]) < 181, lap_pcds))
                    path_list.extend(gau_pcds)
                    path_list.extend(lap_pcds)
        return path_list

    def _get_data_from_path(self, path):

        path_suffix = path.split('/')[-4:]
        path = osp.join(self.root_path,
                        path_suffix[0],
                        path_suffix[1],
                        path_suffix[2],
                        path_suffix[3])

        points = np.fromfile(path, dtype=np.float32)
        points = points.reshape(-1, 4)
        part_labels = points[:, 3]
        # nose_gt = self._get_gt_nose(points[:, 0:3], path)

        # Apply data augmentation
        # add car and persons
        remove_ground = True
        yaw = self._get_yaw(path)
        points, nose_gt, flag = augmentor(points[:, :3], yaw, remove_ground)  # orient at [1, 0, 0], x-axis

        # rotation in x, y and z axes.
        angle = random.randrange(0, 360)
        angle2 = random.randrange(-4, 4)
        angle3 = random.randrange(-4, 4)
        points = PointsProcess(points=points).rotation(pitch=angle2, roll=angle3, yaw=angle).astype(np.float32)  # (N, 3)
        nose_gt = PointsProcess(points=nose_gt).rotation(pitch=angle2, roll=angle3, yaw=angle).astype(np.float32)
        main_axis = PointsProcess(points=np.array([1, 0, 0])).rotation(pitch=angle2, roll=0, yaw=angle).astype(np.float32)

        type = self._get_type(path)

        noise_labels = np.ones(points.shape[0] - part_labels.shape[0], dtype=np.uint16)
        part_labels = np.concatenate((part_labels, noise_labels))
        if remove_ground:
            points = points[flag]
            part_labels = part_labels[flag]
        points = np.concatenate((points, part_labels[:, None]), axis=1)  # (N, 4)

        return type, points, nose_gt, main_axis

    def __getitem__(self, item):
        '''
        The coordinate of simulated point clouds is that a-axis points to the forward,
        y-axis to the up and z-axis to the left.
        Please note that the difference between simulated and real-scanned point clouds
        :param item:
        :return: plane_pc: point cloud of plane, of shape [N, 4]
                 plane_type: category of plane, (A320-200 or B737-200, ...)
                 note-gt:
                 main_axis:
        '''

        inst_path = self.path_list[item]

        # pos_path = self._get_pos(inst_path)
        # neg_path = self._get_neg(inst_path)

        plane_type, plane_pc, nose_gt, main_axis = self._get_data_from_path(inst_path)
        # _, pos = self._get_data_from_path(pos_path)
        # _, neg = self._get_data_from_path(neg_path)

        n = plane_pc.shape[0]
        try:
            idx = np.random.choice(n, 256, replace=True)
        except:
            print(inst_path)
            raise FileNotFoundError

        plane_pc = plane_pc[idx]
        # out = torch.stack([plane_pc, pos, neg])

        plane = torch.zeros(10).cuda()
        plane[plane_type] = 1
        part_labels = plane_pc[:, 3]
        # plane_type = soften(plane_type, v=0.8)  # soft label
        # seg_gt = parts_gt(part_labels=part_labels)
        return plane_pc, plane_type, nose_gt, main_axis, part_labels


if __name__ == '__main__':

    temp_path = '../../../planeSet_crop/data3/B757-200/B757-2000gaussian'

    pcds_path = sorted(glob.glob(osp.join(temp_path, '*noisy*.bin')))

    for pcd in pcds_path:
        points = np.fromfile(pcd, dtype=np.float32)
        points = points.reshape(-1, 4).T
        vis(points[:3, :])
