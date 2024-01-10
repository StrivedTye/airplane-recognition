import os
import numpy as np
import torch.utils.data
import mayavi.mlab as mlab
from test.TestPartSegmentation import segment

from util.point_cloud import PointsProcess
from util import provider
from util.visualization import pic2video
import model.net as MODEL


class TestNoseLocation:
    def __init__(self, model, simulation=False, device="cuda:0"):

        self.model = model
        self.simulation = simulation
        self.device = device
        self.scene_lst = []
        self.predicted_head_lst = []
        self.predicted_nose_lst = []
        self.predicted_body_lst = []
        self.predicted_engine_lst = []
        self.predicted_wing_lst = []
        self.predicted_tail_lst = []
        self.predicted_noise_lst = []

        if self.simulation:
            print('We now test on *Simulation Data*')
            self.nose_error_list = []
        self.line_1 = np.array([[-20 + i * 0.2 - 1.5, -7.4, -10] for i in range(200)])
        self.line_2_mask = np.array(([True] * 10 + [False] * 10) * 40)
        self.line_2 = np.array([[-1.5, -7.4, -10 - j * 0.2] for j in range(800)])[self.line_2_mask]
        self.line = PointsProcess(points=np.concatenate((self.line_1, self.line_2))).rotation(yaw=90)

    @staticmethod
    def _compute_nose_gt(bin_file):
        """
        Only for simulation data
        """
        if not bin_file:
            raise FileNotFoundError("Simulation Mode requires a .bin file for update")

        # load .bin file
        data_np = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        data = torch.from_numpy(data_np).cuda()
        points, part_labels = data[:, :3], data[:, 3:4]
        frame_idx = int(bin_file[-7:-4])

        # load main axis and compute gt nose location
        yaw_list = os.path.dirname(bin_file) + '/pos.txt'
        yaw = np.loadtxt(yaw_list, delimiter=',')[frame_idx - 1][-1] / 180 * np.pi
        yaw_vector = -np.array([np.cos(yaw), 0, np.sin(yaw)])
        points_np = points.cpu().numpy()
        nose_gt = points_np[np.argmax(np.dot(points_np, yaw_vector))]
        return nose_gt

    def test_location(self, points, margin=0.8, cluster=True, nose_gt=None):

        # segment the head out
        part_seg = segment(points=points, model=self.model, device=self.device)
        predicted_head = part_seg['head']
        predicted_body = part_seg['body']
        predicted_engine = part_seg['engine']
        predicted_wing = part_seg['wing']
        predicted_tail = part_seg['tail']
        predicted_noise = part_seg['outlier']


        # filter outliers
        if cluster:
            grav_center = np.expand_dims(np.mean(predicted_head, axis=0), axis=0).repeat(predicted_head.shape[0],
                                                                                         axis=0)
            cluster_mask = np.linalg.norm(predicted_head - grav_center, axis=1) < 2
            predicted_head = predicted_head[cluster_mask]

        # compute predicted nose
        if len(predicted_head):
            predicted_nose = predicted_head[np.argmax(np.dot(predicted_head, np.array([-1, 0, 0])))]
        else:
            raise ValueError('No head has been segmented out!')

        # For simulation data, we additionally compute errors according to gt nose
        if self.simulation:
            nose_error = np.linalg.norm(predicted_nose - nose_gt)
            return predicted_nose, predicted_head, nose_error
        else:
            return predicted_nose, predicted_head, predicted_body, predicted_engine, predicted_wing, predicted_tail, predicted_noise

    def update(self, plane, scene, bin_file=None):

        if self.simulation:
            nose_gt = self._compute_nose_gt(bin_file)  # add the gt nose to the list
            predicted_nose, predicted_head, nose_error = self.test_location(points=plane, nose_gt=nose_gt)
            self.nose_error_list.append(nose_error)
        else:
            predicted_nose, predicted_head, predicted_body, predicted_engine, predicted_wing, predicted_tail, predicted_noise = self.test_location(points=plane)

        self.scene_lst.append(scene)
        self.predicted_nose_lst.append(predicted_nose)
        self.predicted_head_lst.append(predicted_head)
        self.predicted_body_lst.append(predicted_body)
        self.predicted_engine_lst.append(predicted_engine)
        self.predicted_wing_lst.append(predicted_wing)
        self.predicted_tail_lst.append(predicted_tail)
        self.predicted_noise_lst.append(predicted_noise)

    def make_a_frame(self, save_root, scene, idx):

        # prepare saving information
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
            # raise FileNotFoundError('The root does not exist!')
        frames_root = save_root + '/demo_frames'
        if not os.path.exists(frames_root):
            os.mkdir(frames_root)
        file_name = frames_root + '/' + str(idx + 10000) + '.png'

        # acquire updated data
        predicted_head, predicted_nose = self.predicted_head, self.predicted_nose
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 640))

        # plot scene (red), head (green) and predicted nose (white)
        mlab.points3d(scene[:, 0],
                      scene[:, 1],
                      scene[:, 2],
                      scale_factor=0.2,
                      colormap='spectral',
                      color=(1, 0, 0),
                      figure=fig)
        mlab.points3d(predicted_head[:, 0],
                      predicted_head[:, 1],
                      predicted_head[:, 2],
                      scale_factor=0.3,
                      colormap='spectral',
                      color=(0, 0.6, 0),
                      figure=fig)
        mlab.points3d(predicted_nose[0],
                      predicted_nose[1],
                      predicted_nose[2],
                      scale_factor=1.0,
                      colormap='spectral',
                      color=(1, 1, 1),
                      figure=fig)

        # plot lines
        if self.simulation:  # for simulation data
            line_1 = np.array([[-20 + i * 0.2 - 1.5, -7.4, -10] for i in range(200)])
            line_2_mask = np.array(([True] * 10 + [False] * 10) * 40)
            line_2 = np.array([[-1.5, -7.4, -10 - j * 0.2] for j in range(800)])[line_2_mask]
        else:  # for real data
            line_1 = PointsProcess(points=np.array([[30, -4.5, -20 + i * 0.2 + 4] for i in range(200)])).rotation(
                pitch=-0.5, roll=0, yaw=-2)
            line_2_mask = np.array(([True] * 10 + [False] * 10) * 40)
            line_2 = PointsProcess(
                points=np.array([[30 + j * 0.2, -4.5, 4] for j in range(800)])[line_2_mask]).rotation(pitch=-0.5,
                                                                                                      roll=0, yaw=-2)

        mlab.points3d(line_1[:, 0],
                      line_1[:, 1],
                      line_1[:, 2],
                      scale_factor=0.3,
                      colormap='spectral',
                      color=(0.47, 0.47, 0.94),
                      figure=fig)
        mlab.points3d(line_2[:, 0],
                      line_2[:, 1],
                      line_2[:, 2],
                      scale_factor=0.3,
                      colormap='spectral',
                      color=(0.47, 0.47, 0.94),
                      figure=fig)
        mlab.view(azimuth=90, elevation=90, distance=150,
                  focalpoint=(predicted_nose[0], predicted_nose[1], predicted_nose[2]))
        mlab.savefig(filename=file_name)
        mlab.clf(figure=fig)

    def animation_generator(self):
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 640))
        scene = self.scene_lst[0]
        predicted_head = self.predicted_head_lst[0]
        predicted_nose = self.predicted_nose_lst[0]
        predicted_body = self.predicted_body_lst[0]
        predicted_engine = self.predicted_engine_lst[0]
        predicted_wing = self.predicted_wing_lst[0]
        predicted_tail = self.predicted_tail_lst[0]
        predicted_noise = self.predicted_noise_lst[0]

        self.s1 = mlab.points3d(scene[:, 0],
                                scene[:, 1],
                                scene[:, 2],
                                scale_factor=0.2,
                                colormap='spectral',
                                color=(1, 0, 0),
                                figure=fig)

        self.s2 = mlab.points3d(predicted_head[:, 0],
                                predicted_head[:, 1],
                                predicted_head[:, 2],
                                scale_factor=0.3,
                                colormap='spectral',
                                color=(0, 0.6, 0),
                                figure=fig)

        self.s3 = mlab.points3d(predicted_head[:, 0],
                                predicted_head[:, 1],
                                predicted_head[:, 2],
                                scale_factor=1.0,
                                colormap='spectral',
                                color=(1, 1, 1),
                                figure=fig)

        self.s4 = mlab.points3d(predicted_head[:,0],
                                predicted_head[:,1],
                                predicted_head[:,2],
                                scale_factor=0.3,
                                colormap='spectral',
                                color=(1.0, 0.85, 0),
                                figure=fig)

        self.s5 = mlab.points3d(predicted_head[:,0],
                                predicted_head[:,1],
                                predicted_head[:,2],
                                scale_factor=0.3,
                                colormap='spectral',
                                color=(1, 0, 1),
                                figure=fig)

        self.s6 = mlab.points3d(predicted_head[:,0],
                                predicted_head[:,1],
                                predicted_head[:,2],
                                scale_factor=0.3,
                                colormap='spectral',
                                color=(0, 1, 1),
                                figure=fig)

        self.s7 = mlab.points3d(predicted_head[:,0],
                                predicted_head[:,1],
                                predicted_head[:,2],
                                scale_factor=0.3,
                                colormap='spectral',
                                color=(0, 0, 1),
                                figure=fig)

        self.s8 = mlab.points3d(predicted_noise[:,0],
                                predicted_noise[:,1],
                                predicted_noise[:,2],
                                scale_factor=0.3,
                                colormap='spectral',
                                color=(0.6, 0.6, 1),
                                figure=fig)

        self.s9 = mlab.points3d(self.line[:, 0],
                                self.line[:, 1],
                                self.line[:, 2],
                                scale_factor=0.3,
                                colormap='spectral',
                                color=(0.47, 0.47, 0.94),
                                figure=fig)

        self.animation()
        mlab.view(azimuth=90, elevation=90, distance=150,
                  focalpoint=(predicted_nose[0], predicted_nose[1], predicted_nose[2]))
        mlab.show()

    @mlab.animate(delay=50)
    def animation(self):
        for i in range(len(self.scene_lst)):
            self.s1.mlab_source.points = self.scene_lst[i]
            self.s2.mlab_source.points = self.predicted_head_lst[i]
            self.s3.mlab_source.points = self.predicted_nose_lst[i]
            self.s4.mlab_source.points = self.predicted_body_lst[i]
            self.s5.mlab_source.points = self.predicted_engine_lst[i]
            self.s6.mlab_source.points = self.predicted_wing_lst[i]
            self.s7.mlab_source.points = self.predicted_tail_lst[i]
            self.s8.mlab_source.points = self.predicted_noise_lst[i]
            yield

    @staticmethod
    def make_demo_from_frames(frames_root, name='./Demo.mp4'):

        # make video according to saved frames
        pic2video(frames_path=frames_root,
                  size=(640, 587),
                  output_dir=name,
                  fps=25)
        print('Demo has been saved to: ' + name)


if __name__ == '__main__':

    # Define Model
    pretrained_path = './checkpoints/best_model.pth'
    net = MODEL.get_model(normal_channel=False).cuda()
    checkpoints = torch.load(pretrained_path)
    net.load_state_dict(checkpoints['model_state_dict'])
    net.eval()
    torch.cuda.set_device(0)

    # init class
    simulation = True
    location = TestNoseLocation(model=net, simulation=simulation)

    # this frame data
    plane = None
    scene = None
    bin_file = None  # only required when computing error for simulation data

    # updated location instance
    location.update(plane=plane, bin_file=bin_file)
    if simulation:
        nose_error_list = location.nose_error_list

    # saving the visualization result
    save_root = ''
    idx = 0
    location.make_a_frame(save_root=save_root, scene=scene, idx=idx)

    # make demo automatically
    frames_root = ''  # the root of saved frames .png file
    name = './xxx.mp4'
    location.make_demo_from_frames(frames_root=frames_root, name=name)
