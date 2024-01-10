import os
import cv2
import time

# make video from images


def pic2video(frames_path, size, output_dir, fps=25):
    """
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


if __name__ == '__main__':
    pic2video(frames_path=r'/home/mark/NAS/Airplane/real_airplane/20201018_202159_DEFAULT-A320-200_visualization',
              size=(640, 587),
              output_dir=r"/home/mark/NAS/Airplane/real_airplane/visualization/" + str(int(time.time())) + ".mp4",
              fps=50)
