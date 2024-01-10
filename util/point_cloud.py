from pyquaternion import Quaternion
import numpy as np


class PointsProcess:
    def __init__(self, points):
        if len(points.shape) == 1:
            self.points = points
        else:
            self.points = points[:, :3]

    def rotation(self, pitch=0., yaw=0., roll=0.):
        """
        1. the definition of three angles may refer to rotation.png in this project.
        2. pitch, roll, yaw are represented in degree.
        """
        pitch, roll, yaw = pitch/180 * np.pi, roll/180 * np.pi, yaw/180 * np.pi
        quat = Quaternion(axis=np.array([1, 0, 0]), radians=pitch) * \
               Quaternion(axis=np.array([0, 1, 0]), radians=yaw) * \
               Quaternion(axis=np.array([0, 0, 1]), radians=roll)

        return np.dot(self.points, quat.rotation_matrix)


if __name__ == '__main__':
    points = np.random.randn(1000, 5)
    rotated = PointsProcess(points=points).rotation(pitch=2, roll=0.5, yaw=0)
    print(rotated.shape)

