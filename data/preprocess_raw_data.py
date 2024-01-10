# Written by Tye, 09/22/2020
# Structure
# -planeSet
#  --data1
#  --data2
#  --data3
#  --data4
#    --A320-200
#      ......
#    --A330-200
#      -- [type]0gaussian
#      -- [type]0laplace
#      -- [type]1gaussian
#      -- [type]1laplace
#         ......
#      -- [type]9gaussian
#      -- [type]9laplace
# label:
# bridge: 1684632162, car: 7496035, floor: 1851878480, person: 1634563432+1701079411, other: 1701079411
# head: 1702063982, body: 2036625250, engine: 1852137292, wing: 1769430860

import glob
import os.path as osp
import open3d as o3d
import numpy as np
import os

NON_PLANE_LABELS = [1684632162, 7496035, 1851878480, 1634563432, 1701079411]
PLANE_LABELS = [1702063982, 2036625250, 1852137292, 1769430860]

if __name__ == '__main__':
    data_root_path = '/media/mark/Elements SE/planeSet'
    saved_path = '/home/mark/mark/planeSet_crop'
    scenes = glob.glob(osp.join(data_root_path, 'data*'))
    # scenes = scenes[-2:]
    print(scenes)
    for scene in sorted(scenes): # different scenes: sunny or rainy; bridge or not
        scene_name = scene.split('/')[-1]
        for plane_type in glob.glob(osp.join(scene, '*')): # plane type: A340-200
            plane_type_name = plane_type.split('/')[-1]
            for i in range(10):  # 0-9

                # get the path of each frame
                gau_path = osp.join(plane_type, plane_type_name + str(i) + 'gaussian', '*.pcd')
                lap_path = osp.join(plane_type, plane_type_name + str(i) + 'laplace', '*.pcd')
                gau_pcds = sorted(glob.glob(gau_path)) # produced by gaussian noisy
                lap_pcds = sorted(glob.glob(lap_path)) # produced by laplace noisy

                # load point cloud using open3d
                for pcd_path in gau_pcds:
                    cur_pcd_name = pcd_path.split('/')[-1] # e.g. scan00002.pcd
                    cur_pcd_name = cur_pcd_name.split('.')[0] # e.g. scan00002

                    pcd = np.loadtxt(pcd_path, skiprows=11) # slow
                    flag = np.in1d(pcd[:, -1], NON_PLANE_LABELS)
                    all_labels = np.unique(pcd[:, -1])
                    plane_points = pcd[~flag]
                    plane_points = np.array(plane_points[:, [0, 1, 2, 4]], dtype=np.float32)


                    # vis(plane_points.T)

                    saved_name = osp.join(saved_path, scene_name, plane_type_name,
                                          plane_type_name + str(i) + 'gaussian')
                    if not osp.exists(saved_name): os.makedirs(saved_name)
                    saved_name = osp.join(saved_name, cur_pcd_name + '.bin')
                    plane_points.tofile(saved_name)
                    print(saved_name, "Done")

                for pcd_path in lap_pcds:
                    cur_pcd_name = pcd_path.split('/')[-1]
                    cur_pcd_name = cur_pcd_name.split('.')[0]

                    pcd = np.loadtxt(pcd_path,skiprows=11)
                    flag = np.in1d(pcd[:, -1], NON_PLANE_LABELS)
                    plane_points = pcd[~flag]
                    plane_points = np.array(plane_points[:, [0, 1, 2, 4]], dtype=np.float32)

                    saved_name = osp.join(saved_path, scene_name, plane_type_name,
                                          plane_type_name + str(i) + 'laplace')
                    if not osp.exists(saved_name): os.makedirs(saved_name)
                    saved_name = osp.join(saved_name, cur_pcd_name + '.bin')
                    plane_points.tofile(saved_name)
                    print(saved_name, "Done")
