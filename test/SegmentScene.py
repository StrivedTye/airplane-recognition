import numpy as np
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter
from util.point_cloud import PointsProcess
import argparse
import yaml


class SegmentScene:
    def __init__(self, config_file):
        """
        :param scene_dir: the location of the tested scene
        :param ground_height: ture ground height
        :param angle: pitch, roll and yaw
        :param begin: the begin frame for airplane segment
        :param end: the end frame for airplane segment
        :param simulation: False and True corresond to real or fake, respectively
        :param has_bridge: False and True corresond to whether there has bridge in the scene
        """
        with open(config_file) as f:
            config = yaml.safe_load(f)
        self.pitch, self.roll, self.yaw = config['scene_seg'].get('angle')

        self.min = np.array([-1000, -1000, -1000])
        self.max = np.array([1000, 1000, 1000])

        self.bridge = None

        for key, value in config['scene_seg'].items():
            setattr(self, key, value)


    def reset_rang(self):
        self.min = np.array([-1000, -1000, -1000])
        self.max = np.array([1000, 1000, 1000])

    def reset(self, file):

        self.bridge = self.get_bridge(file)

    def filter_rang(self, pcd, rang):
        locate = np.all(pcd[:, :3] > rang[1], axis=1) & np.all(pcd[:, :3] < rang[0], axis=1)
        return pcd[locate], pcd[~locate]

    def filter_outlier(self, pcd):
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        db.fit(pcd)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        lables = db.labels_
        count = Counter(lables)
        lab = [i for i in count if count.get(i) > self.min_nums]
        try:
            lab.remove(-1)
        except BaseException:
            pass
        class_member_mask = np.zeros_like(lables)
        for i in lab:
            class_member_mask = np.logical_or(class_member_mask, (lables == i))
        mask = class_member_mask | core_samples_mask
        return pcd[~mask], pcd[mask]

    def get_bridge(self, pcd):
        if self.simulation:
            pcd[:, [1, 2]] = pcd[:, [2, 1]]
        else:
            pcd = PointsProcess(points=pcd).rotation(pitch=self.pitch, roll=self.roll, yaw=self.yaw)
        z = pcd[:, 2]
        gt_high = z.mean() + self.gt_ref
        pcd = pcd[z > gt_high]
        _, pcd = self.filter_outlier(pcd)
        Z = linkage(pcd, 'single')
        clusters = fcluster(Z, self.cluster_distance, criterion='distance')
        count = Counter(clusters)
        n = max(count, key=count.get)
        bridge = pcd[clusters == n]
        if self.simulation:
            self.ground_height = z.mean() + 0.1
        return np.vstack((bridge.max(axis=0) + 1, bridge.min(axis=0) - 1))

    def get_airplane(self, pcd):
            z = pcd[:, 2]
            air = pcd[z > z.min() + self.cut_height]
            self.min = np.maximum(self.min, air.min(axis=0)) - 2
            self.max = np.minimum(self.max, air.max(axis=0)) + 2
            locate = np.all(pcd > self.min, axis=1) & np.all(pcd < self.max, axis=1)
            airplane = pcd[locate]
            self.min, self.max = airplane.min(axis=0) - np.asarray(self.enlarge_min), airplane.max(axis=0) + np.asarray(self.enlarge_max)
            area = airplane.max(axis=0) - airplane.min(axis=0)
            assert self.area_min <= area[0] * area[1] <= self.area_max
            return airplane, pcd[~locate]

    def forward(self, scene):
        """
        :param scene: point cloud of the input frame
        :return: outlier: noise points, of shape [N, 3]
        :return: bridge: bridge points, of shape [N, 3]
        :return: airplane: airplane points, of shape [N, 3]
        :return: pcd: other points, of shape [N, 3]
        """

        if self.simulation:
            scene[:, [1, 2]] = scene[:, [2, 1]]
            scene = PointsProcess(points=scene).rotation(roll=90)
        else:
            scene = PointsProcess(points=scene).rotation(pitch=self.pitch, roll=self.roll, yaw=self.yaw)

        # crop ground
        pcd = scene[scene[:, 2] > self.ground_height]

        # crop bridge and filter outlier
        if self.has_bridge:
            bridge, pcd = self.filter_rang(pcd, self.bridge)
            outlier, pcd = self.filter_outlier(pcd)


        # get airplane
        try:
            airplane, pcd = self.get_airplane(pcd)
            assert airplane.shape[0] >= 200
        except:
            self.reset_rang()
            return None, None, None, None, None

        # unify axis
        if self.has_bridge:
            outlier[:, [1, 2]] = outlier[:, [2, 1]]
            bridge[:, [1, 2]] = bridge[:, [2, 1]]
        else:
            outlier = None
            bridge = None
        airplane[:, [1, 2]] = airplane[:, [2, 1]]
        pcd[:, [1, 2]] = pcd[:, [2, 1]]
        scene[:,[1,2]] = scene[:,[2,1]]
        return scene, outlier, bridge, airplane, pcd
