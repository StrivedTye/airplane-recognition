# written by Tye, 2020/12/03.
# the main entrance for testing

import os
import numpy as np
import open3d as o3d
import glob
import argparse
import logging
import torch

import model.net as model
from test.SegmentScene import SegmentScene
from test.TestClassification import TestClassification
from test.TestNoseLocation import TestNoseLocation
import yaml


def parse_args():
    parser = argparse.ArgumentParser('Airplane cls and loc')
    parser.add_argument("--config", type=str, default="./config/test.yaml", help="config file")
    return parser.parse_args()


def eval_sequence(seq_path, gt_type, seg_scene, cls_airplane, loc_airplane_nose):
    files = sorted(glob.glob(seq_path + "/*.pcd"))

    # for each frame
    for i, file in enumerate(files):

        if i < seg_scene.begin or (not seg_scene.end==0 and i > seg_scene.end):
            continue

        scene = np.array(o3d.io.read_point_cloud(file).points)

        # we need to reset before recognizing and locating
        if i == seg_scene.begin:
            seg_scene.reset(scene.copy())
            cls_airplane.reset()

        # cls
        scene, _, _, plane, _ = seg_scene.forward(scene.copy())
        if plane is None:
            continue

        cur_pred_ID, cur_pred_type = cls_airplane.predict_type(plane)
        cls_airplane.update(cur_pred_type)
        logging.info("frame=%d, pred_ID=%d, pred_type=%s" % (i, cur_pred_ID, cur_pred_type))

        # loc
        loc_airplane_nose.update(plane=plane, scene=scene)
    loc_airplane_nose.animation_generator()

    pred_type, result = cls_airplane.voting(gt_type)
    logging.info("The predicted airplane type is {}, and it is {}".format(pred_type, result))


if __name__ == "__main__":

    # define args
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('[%s]:' % (k), v)
    print("**************************\n")

    # log
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, 'test.log'),
                        level=logging.INFO)

    # which device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Using device:', device)

    # define model
    net = model.get_model(model=args.backbone, normal_channel=False).to(device)
    checkpoints = torch.load(args.checkpoints)
    net.load_state_dict(checkpoints['model_state_dict'])
    net.eval()

    # scene segmentation
    seg_scene = SegmentScene(args.config)
    # airplane classification
    cls_airplane = TestClassification(model=net, device=device)

    # airplane nose location
    loc_airplane_nose = TestNoseLocation(model=net, device=device, simulation=False)

    # run
    if args.batch_test:

        if args.batch_dir is None:
            raise ValueError("Please specify a path including sequences!")

        seq_paths = sorted(glob.glob(args.batch_dir))

        # for each sequence
        for seq_path in seq_paths:
            files_dir = sorted(glob.glob(seq_path + "/*.pcd"))
            gt_type = ""

            logging.info("===============================================")
            logging.info("Testing sequence: {}".format(seq_path))
            logging.info("Its ground-truth is {}".format(gt_type))
            eval_sequence(seq_path, gt_type, seg_scene, cls_airplane, loc_airplane_nose)
    else:
        logging.info("===============================================")
        logging.info("Testing sequence: {}".format(args.seq_path))
        logging.info("Its ground-truth is {}".format(args.gt_type))

        eval_sequence(args.seq_path, args.gt_type, seg_scene, cls_airplane, loc_airplane_nose)
