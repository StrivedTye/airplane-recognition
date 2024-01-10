# Written by Tye, 22/09/2020
import os
import cv2
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pic2video(frames_path, size, output_dir, fps=25):
    """
    written by Mark, 02/12/2020.
    make video from images
    :param path: file path
    :param size: image_size
    :return:
    """
    # acquire all files in the path
    start_frame = 0
    assert os.path.exists(frames_path)
    filelist = sorted(os.listdir(frames_path))[start_frame:]
    print('Total Frames: ', len(filelist))
    assert len(filelist)

    # different video has different encoding, e.g. 'I','4','2','0' is .avi
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_dir, fourcc, fps, size)

    for item in filelist:
        # whether the file ends with .png
        if item.endswith('.png'):
            item = frames_path + '/' + item
            # read with opencv, channels=BGR, 0-255
            img = cv2.imread(item)
            # write the image into the video
            video.write(img)
        else:
            print('no png file!')

    video.release()


def vis_with_boxes(points, boxes=None, gt_box=None, pause=0.001):
    """
    written by Tye, 22/09/2020.
    :param points: point cloud, of shape [3, N]
    :param boxes: bounding boxes
    :param gt_box: ground-truth box
    :param pause: the show time of every frame
    :return: None
    """

    # Create figure for TRACKING
    fig = plt.figure(1, figsize=(5, 3), facecolor="white")
    plt.rcParams['savefig.dpi'] = 320
    plt.rcParams['figure.dpi'] = 320

    # Create axis in 3D
    ax = fig.gca(projection='3d')

    # Scatter plot the cropped point cloud
    ratio = 1
    ax.scatter(points[0, ::ratio],
               points[1, ::ratio],
               points[2, ::ratio],
               s=3,
               c=points[2, ::ratio])

    # point order to draw a full Box
    order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]

    # Plot GT box
    if gt_box is not None:
        ax.plot(gt_box.corners()[0, order],
                gt_box.corners()[1, order],
                gt_box.corners()[2, order] / 2 - 1,
                color='red',
                alpha=0.5,
                linewidth=5,
                linestyle="-")

    if boxes is not None:
        for id, box in enumerate(boxes):
            box = box.corners()
            ax.plot(box[0, order],
                    box[1, order],
                    box[2, order] / 2 - 1,
                    color='blue',
                    alpha=0.5,
                    linewidth=2,
                    linestyle="-")
    ax.view_init(0, 90)
    # ax.view_init(30, 60)
    plt.axis('off')
    plt.ion()
    plt.show()
    plt.pause(pause)
    plt.clf()


def vis_nose_location(points, predict_head, info):
    """
    added by Mark, 2020.12.2 10:38
    """
    points = points.cpu().numpy()
    predict_head = points[predict_head == 1].cpu().numpy()
    # predict_head = self._remove_outliers(predict_head)

    # visualize
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 640))
    mlab.points3d(points[:, 0],
                  points[:, 1],
                  points[:, 2],
                  mode="point",
                  colormap='spectral', # 'bone', 'copper', 'gnuplot'
                  color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                  figure=fig,
                  )
    mlab.points3d(predict_head[:, 0],
                  predict_head[:, 1],
                  predict_head[:, 2],
                  scale_factor=0.3,
                  colormap='spectral', # 'bone', 'copper', 'gnuplot'
                  color=(1, 0, 0),   # Used a fixed (r,g,b) instead
                  figure=fig,
                  )
    mlab.points3d(info[0],
                  info[1],
                  info[2],
                  scale_factor=0.7,
                  colormap='spectral',
                  color=(1, 1, 0),
                  figure=fig,
                  )
    mlab.points3d(info[3],
                  info[4],
                  info[5],
                  scale_factor=0.7,
                  colormap='spectral',
                  color=(1, 0, 1),
                  figure=fig,
                  )
    mlab.show()

def vis_seg_scene(pcd):
    mlab.points3d(pcd[:, 0], pcd[:, 1], pcd[:, 2], color=(1, 0, 0), scale_factor=0.5)
    mlab.points3d(pcd[:, 0], pcd[:, 1], np.zeros_like(pcd[:, 2]) + pcd[:, 2].mean() - 1.8, color=(0, 1, 0),
                  scale_factor=0.5)
    mlab.show()


def mlabvis(pcd, pcd2=None, pcd3=None, pcd4=None, pcd5=None, pcd6=None):
        print('')
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 640))
        mlab.points3d(pcd[:, 0], pcd[:, 1], pcd[:, 2], color=(1, 0, 0), scale_factor=0.5, figure=fig)
        if pcd2 is not None:
            mlab.points3d(pcd2[:, 0], pcd2[:, 1], pcd2[:, 2], color=(0, 1, 0), scale_factor=1.0, figure=fig)
        # mlab.points3d(pcd[:, 0], pcd[:, 1], np.zeros_like(pcd[:, 2]) + pcd[:, 2].mean(), color=(0, 1, 0),
        # scale_factor=0.5)
        if pcd3 is not None:
            mlab.points3d(pcd3[:, 0], pcd3[:, 1], pcd3[:, 2], color=(0, 0, 1), scale_factor=1.0, figure=fig)
        if pcd4 is not None:
            mlab.points3d(pcd4[:, 0], pcd4[:, 1], pcd4[:, 2], color=(1, 0, 1), scale_factor=1.0, figure=fig)
        if pcd5 is not None:
            mlab.points3d(pcd5[:, 0], pcd5[:, 1], pcd5[:, 2], color=(0, 1, 1), scale_factor=1.0, figure=fig)
        if pcd6 is not None:
            mlab.points3d(pcd6[:, 0], pcd6[:, 1], pcd6[:, 2], color=(1, 1, 1), scale_factor=1.0, figure=fig)
        mlab.show()
