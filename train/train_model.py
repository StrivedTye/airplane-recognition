import os
import datetime
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.append('../')
from data.dataset import PlaneDataset
from util import provider
import model.net as MODEL
from util.point_cloud import PointsProcess
import warnings
warnings.filterwarnings('ignore')
from data.parts_gt import parts_gt, scale_loss


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training [default: 2]')
    parser.add_argument('--train', type=bool, default=True, help='whether train or test [default: False]')
    parser.add_argument('--axis', type=bool, default=False, help='batch size in training [default: 2]')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--model', default='pointnet', choices=['pointnet', 'DGCNN'], help='model name [default: pointnet_cls]')
    parser.add_argument('--data_dir', type=str, default='/media/mark/Elements SE/planeSet_crop', help='dataset path')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()


def test(model, loader, num_class=10):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader, 0), total=len(loader)):
        data_, target, nose_gt, main_axis, _ = data  # (128, 256, 4)
        points = data_[:, :, :3].cpu().numpy()
        part_labels = data_[:, :, 3:4]
        head_gt = ((part_labels.squeeze() - torch.ones(128, 256)*1702063982) == 0).float()
        points = provider.normalize_data(points)
        points = torch.Tensor(points)[:, :, :3]
        points = points.transpose(2, 1)
        points, target, head_gt, main_axis = points.cuda(), target.cuda(), head_gt.cuda(), main_axis.cuda()

        classifier = model.eval()
        pred, _, pred_head, pred_main_axis, scale, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        # target = torch.argmax(target, dim=-1)
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat, 0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat, 1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.train:
        '''CREATE DIR'''
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = Path('./log/')
        if not os.path.exists(experiment_dir):
            experiment_dir.mkdir(exist_ok=True)
        experiment_dir = experiment_dir.joinpath('classification')
        if not os.path.exists(experiment_dir):
            experiment_dir.mkdir(exist_ok=True)
        if args.log_dir is None:
            experiment_dir = experiment_dir.joinpath(timestr)
        else:
            experiment_dir = experiment_dir.joinpath(args.log_dir)
        experiment_dir.mkdir(exist_ok=True)
        checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
        log_dir = experiment_dir.joinpath('logs/')
        log_dir.mkdir(exist_ok=True)

        '''LOG'''
        args = parse_args()
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        log_string('PARAMETER ...')
        log_string(args)

        '''DATA LOADING'''
        log_string('Load dataset ...')

        train_data = PlaneDataset(root_path=args.data_dir, split='train')
        trainDataLoader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=int(args.workers),
                                                       pin_memory=False)

        test_data = PlaneDataset(root_path=args.data_dir, split='test')
        testDataLoader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=int(args.workers),
                                                  pin_memory=True)
        print('#Train data:', len(train_data), '#Test data:', len(test_data))

        '''MODEL LOADING'''
        num_class = 10
        classifier = MODEL.get_model(num_cls=num_class, model=args.model, normal_channel=args.normal).cuda()
        criterion = MODEL.get_loss().cuda()
        # mask_mse = nn.L1Loss().cuda()
        mask_mse = nn.MSELoss().cuda()
        CE = nn.CrossEntropyLoss().cuda()
        L1 = nn.SmoothL1Loss().cuda()

        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
        except:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0


        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        global_epoch = 0
        global_step = 0
        best_instance_acc = 0.0
        best_class_acc = 0.0
        mean_correct = []

        '''TRANING'''
        logger.info('Start training...')
        for epoch in range(start_epoch,args.epoch):
            log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
            mean = 0
            mean_gt = 0

            scheduler.step()
            for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
                data_, target, nose_gt, main_axis, part_labels = data  # (128, 256, 4), nose_gt = [128, 3]
                points = data_[:, :, :3].cpu().numpy()
                # target = data_[:, :, 3:4]
                # target = target[:, 0]
                seg_gt = parts_gt(part_labels=part_labels)
                points = provider.normalize_data(points)
                points = provider.shift_point_cloud(points[:, :, 0:3])
                points = torch.Tensor(points)[:, :, :3]
                points = points.transpose(2, 1)  # [128, 3, 256]
                points, target, main_axis = points.cuda(), target.cuda(), main_axis.cuda()
                optimizer.zero_grad()

                classifier = classifier.train()
                pred, trans_feat, pred_seg, pred_main_axis, scale, ab = classifier(points)
                # loss_cls = L1(input=pred, target=target.float())
                # target = torch.argmax(target, dim=-1)
                loss_cls = criterion(pred, target.long(), trans_feat)
                loss_ab = CE(input=ab, target=(target < 4).long())
                # loss_scale = scale_loss(target=target, pred_distance=scale)
                loss_seg = CE(input=pred_seg.transpose(2, 1).reshape(128*256, 6), target=seg_gt)
                loss_axis = mask_mse(input=pred_main_axis, target=main_axis)
                loss = loss_cls + 0.5 * loss_seg + 0.2 * loss_ab + loss_axis
                print('cls', loss_cls.item(), 'seg', 0.5 * loss_seg.item(), 'ab', 0.2 * loss_ab.item(), 'axis', loss_axis.item(), 'loss', loss.item())

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))
                loss.backward()
                optimizer.step()
                global_step += 1

            train_instance_acc = np.mean(mean_correct)
            log_string('Train Instance Accuracy: %f' % train_instance_acc)

            with torch.no_grad():
                instance_acc, class_acc = test(classifier.eval(), testDataLoader)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc
                log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

                if (instance_acc >= best_instance_acc):
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s'% savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    print('Epoch:', global_epoch, 'best_model has been saved')
                    torch.save(state, savepath)
                global_epoch += 1

        logger.info('End of training...')
    else:
        test_data = PlaneDataset(root_path=args.data_dir, split='test')
        testDataLoader = torch.utils.data.DataLoader(test_data,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=int(args.workers),
                                                     pin_memory=True)
        print('#Test data:', len(test_data))
        num_class = 10
        classifier = MODEL.get_model(k = num_class, backbone=args.model, normal_channel=args.normal).cuda()
        checkpoint = torch.load( './log/classification/2020-10-29_15-08/checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)
            print('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)

