import numpy as np
# import mayavi.mlab as mlab
from util.point_cloud import PointsProcess
import random
import scipy

NON_PLANE_LABELS = [1684632162, 7496035, 1851878480, 1634563432, 1701079411, 1936679753]
PLANE_LABELS = [1702063982, 2036625250, 1852137292, 1769430860]


# def mlabvis(pcd, nose_gt):
#     mlab.points3d(pcd[:, 0], pcd[:, 1], pcd[:, 2], scale_factor=0.2, color=(1, 0, 0))
#     mlab.points3d(nose_gt[0], nose_gt[1], nose_gt[2], scale_factor=0.7, color=(0, 1, 0))
#     mlab.show()


def random_noise(pcd):
    '''
    以 70% 的概率为飞机随机添加噪声
    策略为： 噪声点个数为0-30之间的随机整数
    x 轴方向（飞机机头朝向）上， 随机在-5 —— 10米内添加噪声
    y 轴方向随机添加
    z 轴方向随机添加在机头-5 —— 5米内
    :param pcd: 飞机点云数组
    :return: 带噪声的飞机点云数据
    '''
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    x_scale = random.random() * 15
    y_scale = y.max() - y.min()
    z_scale = random.random() * 10
    num = random.randint(0, 30)
    noise = np.vstack((np.random.random(num) * x_scale + x.max() - x_scale / 3,
                       np.random.random(num) * y_scale +  y.min(),
                       np.random.random(num) * z_scale + z.mean() - z_scale / 2))
    noise = np.transpose(noise)
    return noise


def random_car(pcd):
    '''
    在飞机上随机添加密集噪声点模拟车和人
    :param pcd:
    :return:
    '''
    num = random.randint(0, 5)
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    ret = []
    for i in range(num):
        scale = random.randint(5, 30)
        car = scipy.rand(scale, 3)
        car_x_scale = random.uniform(0.2, 1)
        car_y_scale = random.uniform(0.6, 2)
        car_z_scale = random.uniform(0.2, 1)
        random_scale = np.array([car_x_scale, car_y_scale, car_z_scale])
        random_pos = np.random.random(3) * np.array([25, 5, 15]) + np.array([x.max()-5, y.min(), z.min()])
        car = np.asarray(car) * random_scale + random_pos
        ret.append(car)
    return ret


def random_remove(pcd):
    x = pcd[:, 0]
    scale = random.uniform(3, 8)
    flag = x > (x.min() + scale)
    return pcd[flag]


def random_rv_ground(pcd):
    higth = np.random.normal(loc=0.6, scale=0.5)
    while higth > 1.7:
        higth = np.random.normal(loc=1, scale=0.5)
    flag = pcd[:, 1] > (pcd[:,1].min() + higth)
    z = pcd[:, 2]
    scale_z = random.uniform(0, 3)
    if random.random() < 0.25:
        flag = flag & (z < z.max() - scale_z)
    return flag


def augmentor(points, yaw, remove_ground=True):
    points = PointsProcess(points=points).rotation(pitch=0, roll=0, yaw=yaw)  # orientation at y-axis
    nose_gt = points[np.argmax(points[:, 0:1])]

    # add noise
    if random.random() < 0.25:
        noise = random_noise(points)

    if random.random() < 0.25:
        cars = random_car(points)

    # if random.random() < 1:
    #     points = random_remove(points)

    if 'noise' in dir():
        points = np.vstack((points, noise))

    if 'cars' in dir():
        for car in cars:
            points = np.vstack((points, car))

    flag = None
    if remove_ground:
        flag = random_rv_ground(points)

    return points, nose_gt, flag


if __name__ == '__main__':
    frame_idx = 110
    posfile = '/media/mark/Elements SE/planeSet/data1/A320-200/A320-2000gaussian/pos.txt'
    file = '/media/mark/Elements SE/planeSet/data1/A320-200/A320-2000gaussian/scan_noisy00111.pcd'
    yaw = np.loadtxt(posfile, delimiter=',')[frame_idx - 1][-1]  # angle
    scan = np.loadtxt(file, skiprows=11)
    flag = np.in1d(scan[:, -1], NON_PLANE_LABELS)
    all_labels = np.unique(scan[:, -1])
    points = scan[~flag]
    points = points[:, :3]
    points = PointsProcess(points=points).rotation(pitch=0, roll=0, yaw=yaw)
    nose_gt = points[np.argmax(points[:, 0:1])]
    # 机头附近随机添加噪声
    if random.random() < 1:
        noise = random_noise(points)
    # 机头附近随机添加密集噪声(模拟车和人)
    if random.random() < 1:
        cars = random_car(points)
    # 随机砍掉部分机尾
    # if random.random() < 1:
    #     points = random_remove(points)

    if 'noise' in dir():
        points = np.vstack((points, noise))
    if 'cars' in dir():
        for car in cars:
            points = np.vstack((points, car))
    points = random_rv_ground(points)
    print(points[:,1].min())
    mlabvis(points, nose_gt)
