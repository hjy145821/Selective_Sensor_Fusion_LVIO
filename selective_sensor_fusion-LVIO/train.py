import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import re
import argparse
import time
import csv
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
from models import FlowNet, RecFeat, PoseRegressor, RecImu, Fc_Flownet, Hard_Mask, Soft_Mask
from utils import save_path_formatter, mat2euler
from logger import AverageMeter
from itertools import chain
from pathlib import Path
import torch.nn.functional as F
from data_loader import KITTI_Loader
from data_loader import Args

parser = argparse.ArgumentParser(description='Selective Sensor Fusion on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

best_error = -1
n_iter = 0

def main():

    global best_error, n_iter
    args = Args()
    # args = parser.parse_args()
    # degradation_mode = 0
    fusion_mode = 6
    # 0: vision only 1: direct 2: soft 3: hard 4: Lvio direct 5: Lvio soft 6: Lvio hard
    # # set saving path
    abs_path = Path('D:\\SLAM')
    save_path = save_path_formatter(args)
    save_path = str(save_path)  
    save_path = re.sub(r'[,:]', '_', save_path) 
    args.save_path = abs_path / 'path' / 'checkpoints' / save_path
    print('=> will save everything to {}'.format(args.save_path))
    if not (args.save_path / 'imgs').exists():
        (args.save_path / 'imgs').mkdir(parents=True)
    if not (args.save_path / 'models').exists():
        (args.save_path / 'models').mkdir(parents=True)

    normalize = custom_transforms.Normalize(mean=[0, 0, 0],
                                            std=[255, 255, 255])
    normalize2 = custom_transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    input_transform = custom_transforms.Compose_imgs([
        custom_transforms.ArrayToTensor(),
        normalize,
        normalize2
    ])
    # Data loading code
    print("=> fetching scenes in '{}'".format(args.data))

    train_set = KITTI_Loader(
        args.data,
        transform=input_transform,
        train=0,
        sequence_length=args.sequence_length,
        fusion_mode=args.fusion_mode,
    )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    val_set = KITTI_Loader(
        args.data,
        transform=input_transform,
        train=1,
        sequence_length=args.sequence_length,
        fusion_mode=args.fusion_mode,
    )
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    test_set = KITTI_Loader(
        args.data,
        transform=input_transform,
        train=2,
        sequence_length=args.sequence_length,
        fusion_mode=args.fusion_mode,
    )
    print('{} samples found in {} test scenes'.format(len(test_set), len(test_set.scenes)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create pose model
    print("=> creating pose model")

    feature_dim = 256
    if fusion_mode == 0:
        feature_ext = FlowNet(args.batch_size).cuda()
        feature_ext_pc = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim*2).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 1:
        feature_ext = FlowNet(args.batch_size).cuda()
        feature_ext_pc = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 2:
        feature_ext = FlowNet(args.batch_size).cuda()
        feature_ext_pc = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Soft_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 3:
        feature_ext = FlowNet(args.batch_size).cuda()
        feature_ext_pc = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 2, feature_dim * 2, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 2, feature_dim * 2).cuda()
        pose_net = PoseRegressor(feature_dim * 2).cuda()

    if fusion_mode == 4:
        feature_ext = FlowNet(args.batch_size).cuda()
        feature_ext_pc = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 3, feature_dim * 3, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 3, feature_dim * 3).cuda()
        pose_net = PoseRegressor(feature_dim * 3).cuda()

    if fusion_mode == 5:
        feature_ext = FlowNet(args.batch_size).cuda()
        feature_ext_pc = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 3, feature_dim * 3, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Soft_Mask(feature_dim * 3, feature_dim * 3).cuda()
        pose_net = PoseRegressor(feature_dim * 3).cuda()

    if fusion_mode == 6:
        feature_ext = FlowNet(args.batch_size).cuda()
        feature_ext_pc = FlowNet(args.batch_size).cuda()
        fc_flownet = Fc_Flownet(32 * 1024, feature_dim).cuda()
        rec_feat = RecFeat(feature_dim * 3, feature_dim * 3, args.batch_size, 2).cuda()
        rec_imu = RecImu(6, int(feature_dim / 2), args.batch_size, 2, feature_dim).cuda()
        selectfusion = Hard_Mask(feature_dim * 3, feature_dim * 3).cuda()
        pose_net = PoseRegressor(feature_dim * 3).cuda()

    pose_net.init_weights()

    flownet_model_path = str(abs_path) + '\\path\\FlowNetPytorch\\pretrain\\Flownets_EPE1.951.pth.tar'
    pretrained_flownet = True
    if pretrained_flownet:
        weights = torch.load(flownet_model_path)
        model_dict = feature_ext.state_dict()
        update_dict = {k: v for k, v in weights['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        feature_ext.load_state_dict(model_dict)
        print('restrore depth model from ' + flownet_model_path)

    cudnn.benchmark = True
    feature_ext = torch.nn.DataParallel(feature_ext)
    rec_feat = torch.nn.DataParallel(rec_feat)
    pose_net = torch.nn.DataParallel(pose_net)
    rec_imu = torch.nn.DataParallel(rec_imu)
    fc_flownet = torch.nn.DataParallel(fc_flownet)
    selectfusion = torch.nn.DataParallel(selectfusion)

    print('=> setting adam solver')

    parameters = chain(rec_feat.parameters(), rec_imu.parameters(), fc_flownet.parameters(), pose_net.parameters(),
                       selectfusion.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path / args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'train_pose', 'train_euler', 'validation_loss', 'val_pose', 'val_euler'])

    # start training loop
    print('=> training pose model')
    print('fusion_mode = ', fusion_mode)

    best_val = 100
    best_tra = 10000.0
    best_ori = 10000.0

    for epoch in range(args.epochs):

        # train for one epoch
        train_loss, pose_loss, euler_loss, temp = train(args, train_loader, feature_ext, feature_ext_pc, rec_feat, rec_imu, pose_net,
                                                        fc_flownet, selectfusion, optimizer, epoch, fusion_mode)
        temp = 0.5

        # evaluate on validation set
        val_loss, val_pose_loss, val_euler_loss =\
            validate(args, val_loader, feature_ext, feature_ext_pc, rec_feat, rec_imu, pose_net, fc_flownet, selectfusion, temp, epoch,
                     fusion_mode)

        test(args, test_loader, feature_ext, feature_ext_pc, rec_feat, rec_imu, pose_net,
             fc_flownet, selectfusion, temp, epoch, fusion_mode)

        if val_pose_loss < best_tra:
            best_tra = val_pose_loss

        if val_euler_loss < best_ori:
            best_ori = val_euler_loss
        save_path
        print('Best: {}, Best Translation {:.5} Best Orientation {:.5}'
              .format(epoch + 1, best_tra, best_ori))

        # save checkpoint
        if (val_loss < best_val):

            best_val = val_loss

            fn = str(args.save_path) + '\\models\\rec_' + str(epoch) + '.pth'
            torch.save(rec_feat.module.state_dict(), fn)

            fn = str(args.save_path) + '\\models\\pose_' + str(epoch) + '.pth'
            torch.save(pose_net.module.state_dict(), fn)

            fn = str(args.save_path) + '\\models\\fc_flownet_' + str(epoch) + '.pth'
            torch.save(fc_flownet.module.state_dict(), fn)

            fn = str(args.save_path) + '\\models\\rec_imu_' + str(epoch) + '.pth'
            torch.save(rec_imu.module.state_dict(), fn)

            fn = str(args.save_path) + '\\models\\selectfusion_' + str(epoch) + '.pth'
            torch.save(selectfusion.module.state_dict(), fn)
            print('Model has been saved')

        with open(str(args.save_path) + '\\' + args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, pose_loss, euler_loss, val_loss, val_pose_loss, val_euler_loss])

def train(args, train_loader, feature_ext, feature_ext_pc, rec_feat, rec_imu, pose_net, fc_flownet, selectfusion, optimizer, epoch,
          fusion_mode):

    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()

    epoch_size = args.epoch_size

    # switch to train mode
    feature_ext.eval()
    feature_ext_pc.eval()
    rec_feat.train()
    rec_imu.train()
    pose_net.train()
    fc_flownet.train()
    selectfusion.train()

    end = time.time()

    aver_loss = 0
    aver_pose_loss = 0
    aver_euler_loss = 0
    aver_n = 0

    temp_min = 0.5
    ANNEAL_RATE = 0.0006
    temp = 1.0

    for i, (imgs, imus, poses, pc_imgs) in enumerate(train_loader):

        if len(imgs[0]) != args.batch_size:
            continue

        # measure data loading time
        data_time.update(time.time() - end)

        rec_feat.module.hidden = rec_feat.module.init_hidden()

        pose_loss = 0
        euler_loss = 0

        # compute output
        for j in range(0, len(imgs)-1):

            tgt_img = imgs[j+1]
            ref_img = imgs[j]
            imu = imus[j]

            if torch.cuda.is_available():
                tgt_img_var = Variable(tgt_img.cuda())
                ref_img_var = Variable(ref_img.cuda())
                imu_var = Variable(imu.transpose(0, 1).cuda())
            else:
                tgt_img_var = Variable(tgt_img)
                ref_img_var = Variable(ref_img)
                imu_var = Variable(imu.transpose(0, 1))

            rec_imu.module.hidden = rec_imu.module.init_hidden()

            with torch.no_grad():

                raw_feature_vision = feature_ext(tgt_img_var, ref_img_var)
                feature_vision = fc_flownet(raw_feature_vision)
                feature_imu = rec_imu(imu_var)

                if args.fusion_mode in [4, 5, 6]:
                    tgt_pc_img = pc_imgs[j+1]
                    ref_pc_img = pc_imgs[j]

                    if torch.cuda.is_available():
                        tgt_pc_img_var = Variable(tgt_pc_img.cuda())
                        ref_pc_img_var = Variable(ref_pc_img.cuda())
                    else:
                        tgt_pc_img_var = Variable(tgt_pc_img)
                        ref_pc_img_var = Variable(ref_pc_img)

                    raw_feature_points = feature_ext_pc(tgt_pc_img_var, ref_pc_img_var)
                    feature_points = fc_flownet(raw_feature_points)
                    feature_pc = torch.cat([feature_vision, feature_points, feature_imu], 2)

                    if fusion_mode == 4:
                        feature_weighted = feature_pc

                    if fusion_mode == 5:
                        mask = selectfusion(feature_pc)
                        feature_weighted = torch.cat([feature_vision, feature_points, feature_imu], 2) * mask

                    elif fusion_mode == 6: 
                        mask = selectfusion(feature_pc, temp)
                        feature_weighted = torch.cat([feature_vision, feature_points, feature_imu], 2) * mask

                else:
                    feature = torch.cat([feature_vision, feature_imu], 2)

                    if fusion_mode == 0:
                        feature_weighted = feature_vision

                    elif fusion_mode == 1:
                        feature_weighted = feature

                    elif fusion_mode == 2:
                        mask = selectfusion(feature)
                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                    elif fusion_mode == 3:
                        mask = selectfusion(feature, temp)
                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                    else:
                        raise ValueError(f"Invalid fusion_mode: {fusion_mode}")

            # recurrent features
            feature_new = rec_feat(feature_weighted)

            # pose net
            pose = pose_net(feature_new)

            # compute pose err
            pose = pose.view(-1, 6)

            trans_pose = compute_trans_pose(poses[j].cpu().data.numpy().astype(np.float64),
                                            poses[j + 1].cpu().data.numpy().astype(np.float64))

            if torch.cuda.is_available():
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1]).cuda()
            else:
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1])

            rot_mat = torch.FloatTensor(trans_pose[:, :, :3]).cuda()

            euler = mat2euler(rot_mat)

            euler_loss += F.mse_loss(euler, pose[:, 3:])

            pose_loss += F.mse_loss(pose_truth, pose[:, :3])

        euler_loss /= (len(imgs) - 1)
        pose_loss /= (len(imgs) - 1)

        loss = pose_loss + euler_loss * 100

        aver_loss += loss.item()
        aver_pose_loss += pose_loss.item()
        aver_euler_loss += euler_loss.item()

        aver_n += 1

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            print('Train: Epoch [{}/{}] Step [{}/{}]: Time {} Data {} Loss {:.5} '
                  'Pose {:.5} Euler {:.5}'.
                  format(epoch + 1, args.epochs, i + 1, epoch_size,
                         batch_time, data_time, loss.item(), pose_loss.item(), euler_loss.item()))

        # decrease hard mask temperature
        if i % 10 == 0:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)

        if i >= epoch_size - 1:
            break

        n_iter += 1
    
    aver_loss /= aver_n
    aver_pose_loss /= aver_n
    aver_euler_loss /= aver_n

    print('Train: {}, Average_Loss {:.5} pose_loss {:.5} euler_loss {:.5}'
          .format(epoch + 1, aver_loss, aver_pose_loss, aver_euler_loss))

    return aver_loss, aver_pose_loss, aver_euler_loss, temp

def validate(args, val_loader, feature_ext, feature_ext_pc, rec_feat, rec_imu, pose_net, fc_flownet, selectfusion, temp, epoch,
             fusion_mode):

    batch_time = AverageMeter()

    # switch to evaluate mode
    feature_ext.eval()
    rec_feat.eval()
    pose_net.eval()
    rec_imu.eval()
    fc_flownet.eval()
    selectfusion.eval()

    end = time.time()

    aver_loss = 0
    aver_pose_loss = 0
    aver_euler_loss = 0
    aver_n = 0

    for i, (imgs, imus, poses, pc_imgs) in enumerate(val_loader):

        if len(imgs[0]) != args.batch_size:
            continue

        rec_feat.module.hidden = rec_feat.module.init_hidden()

        pose_loss = 0
        euler_loss = 0

        # compute output
        for j in range(0, len(imgs) - 1):

            tgt_img = imgs[j + 1]
            ref_img = imgs[j]
            imu = imus[j]

            if torch.cuda.is_available():
                tgt_img_var = Variable(tgt_img.cuda())
                ref_img_var = Variable(ref_img.cuda())
                imu_var = Variable(imu.transpose(0, 1).cuda())
            else:
                tgt_img_var = Variable(tgt_img)
                ref_img_var = Variable(ref_img)
                imu_var = Variable(imu.transpose(0, 1))

            with torch.no_grad():
                rec_imu.module.hidden = rec_imu.module.init_hidden()
                raw_feature_vision = feature_ext(tgt_img_var, ref_img_var)
                feature_vision = fc_flownet(raw_feature_vision)
                feature_imu = rec_imu(imu_var)

                if args.fusion_mode in [4, 5, 6]:
                    tgt_pc_img = pc_imgs[j+1]
                    ref_pc_img = pc_imgs[j]

                    if torch.cuda.is_available():
                        tgt_pc_img_var = Variable(tgt_pc_img.cuda())
                        ref_pc_img_var = Variable(ref_pc_img.cuda())
                    else:
                        tgt_pc_img_var = Variable(tgt_pc_img)
                        ref_pc_img_var = Variable(ref_pc_img)

                    raw_feature_points = feature_ext_pc(tgt_pc_img_var, ref_pc_img_var)
                    feature_points = fc_flownet(raw_feature_points)
                    feature_pc = torch.cat([feature_vision, feature_points, feature_imu], 2)

                    if fusion_mode == 4:
                        feature_weighted = feature_pc

                    if fusion_mode == 5:
                        mask = selectfusion(feature_pc)
                        feature_weighted = torch.cat([feature_vision, feature_points, feature_imu], 2) * mask

                    elif fusion_mode == 6:
                        mask = selectfusion(feature_pc, temp)
                        feature_weighted = torch.cat([feature_vision, feature_points, feature_imu], 2) * mask

                else:
                    feature = torch.cat([feature_vision, feature_imu], 2)

                    if fusion_mode == 0:
                        feature_weighted = feature_vision

                    elif fusion_mode == 1:
                        feature_weighted = feature

                    elif fusion_mode == 2:
                        mask = selectfusion(feature)
                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                    elif fusion_mode == 3:
                        mask = selectfusion(feature, temp)
                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                    else:
                        raise ValueError(f"Invalid fusion_mode: {fusion_mode}")

                
                # recurrent features
                feature_new = rec_feat(feature_weighted)

                pose = pose_net(feature_new)

            # compute pose err
            pose = pose.view(-1, 6)

            trans_pose = compute_trans_pose(poses[j].cpu().data.numpy().astype(np.float64),
                                            poses[j + 1].cpu().data.numpy().astype(np.float64))

            if torch.cuda.is_available():
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1]).cuda()
            else:
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1])

            rot_mat = torch.FloatTensor(trans_pose[:, :, :3]).cuda()

            euler = mat2euler(rot_mat)

            euler_loss += F.mse_loss(euler, pose[:, 3:])

            pose_loss += F.mse_loss(pose_truth, pose[:, :3])

        euler_loss /= (len(imgs) - 1)
        pose_loss /= (len(imgs) - 1)

        loss = pose_loss + euler_loss * 100

        aver_pose_loss += pose_loss.item()
        aver_loss += loss.item()

        aver_euler_loss += euler_loss.item()

        aver_n += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Val: Epoch [{}/{}] Step [{}/{}]: Loss {:.5} '
                  'Pose {:.5} Euler {:.5}'.
                  format(epoch + 1, args.epochs, i + 1, len(val_loader), loss.item(), pose_loss.item(),
                         euler_loss.item()))

    aver_loss /= aver_n
    aver_pose_loss /= aver_n
    aver_euler_loss /= aver_n
    print('Val: {}, Average_Loss {:.5} Pose_loss {:.5} Euler_loss {:.5}'
          .format(epoch + 1, aver_loss, aver_pose_loss, aver_euler_loss))

    return aver_loss, aver_pose_loss, aver_euler_loss

def test(args, test_loader, feature_ext, feature_ext_pc, rec_feat, rec_imu, pose_net,
         fc_flownet, selectfusion, temp, epoch, fusion_mode):

    batch_time = AverageMeter()

    # switch to evaluate mode
    feature_ext.eval()
    rec_feat.eval()
    pose_net.eval()
    rec_imu.eval()
    fc_flownet.eval()
    selectfusion.eval()

    end = time.time()

    aver_loss = 0
    aver_pose_loss = 0
    aver_euler_loss = 0
    aver_n = 0

    for i, (imgs, imus, poses, pc_imgs) in enumerate(test_loader):

        if i == 0:
            k = 7
        if i == 1:
            k = 10

        result = []
        truth_pose = []
        truth_euler = []

        rec_feat.module.hidden = rec_feat.module.init_test_hidden()

        pose_loss = 0
        euler_loss = 0

        # compute output
        for j in range(0, len(imgs) - 1):

            tgt_img = imgs[j + 1]
            ref_img = imgs[j]
            imu = imus[j]

            if torch.cuda.is_available():
                tgt_img_var = Variable(tgt_img.cuda())
                ref_img_var = Variable(ref_img.cuda())
                imu_var = Variable(imu.transpose(0, 1).cuda())
            else:
                tgt_img_var = Variable(tgt_img)
                ref_img_var = Variable(ref_img)
                imu_var = Variable(imu.transpose(0, 1))

            with torch.no_grad():
                
                rec_imu.module.hidden = rec_imu.module.init_test_hidden()
                raw_feature_vision = feature_ext(tgt_img_var, ref_img_var)
                feature_vision = fc_flownet(raw_feature_vision)
                feature_imu = rec_imu(imu_var)

                if args.fusion_mode in [4, 5, 6]:
                    tgt_pc_img = pc_imgs[j+1]
                    ref_pc_img = pc_imgs[j]

                    if torch.cuda.is_available():
                        tgt_pc_img_var = Variable(tgt_pc_img.cuda())
                        ref_pc_img_var = Variable(ref_pc_img.cuda())
                    else:
                        tgt_pc_img_var = Variable(tgt_pc_img)
                        ref_pc_img_var = Variable(ref_pc_img)

                    raw_feature_points = feature_ext_pc(tgt_pc_img_var, ref_pc_img_var)
                    feature_points = fc_flownet(raw_feature_points)
                    feature_pc = torch.cat([feature_vision, feature_points, feature_imu], 2)

                    if fusion_mode == 4:
                        feature_weighted = feature_pc

                    if fusion_mode == 5:
                        mask = selectfusion(feature_pc)
                        feature_weighted = torch.cat([feature_vision, feature_points, feature_imu], 2) * mask

                    elif fusion_mode == 6:
                        mask = selectfusion(feature_pc, temp)
                        feature_weighted = torch.cat([feature_vision, feature_points, feature_imu], 2) * mask

                else:
                    feature = torch.cat([feature_vision, feature_imu], 2)

                    if fusion_mode == 0:
                        feature_weighted = feature_vision

                    elif fusion_mode == 1:
                        feature_weighted = feature

                    elif fusion_mode == 2:
                        mask = selectfusion(feature)
                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                    elif fusion_mode == 3:
                        mask = selectfusion(feature, temp)
                        feature_weighted = torch.cat([feature_vision, feature_imu], 2) * mask

                    else:
                        raise ValueError(f"Invalid fusion_mode: {fusion_mode}")
                
                # recurrent features
                feature_new = rec_feat(feature_weighted)

                pose = pose_net(feature_new)

            # compute pose err
            pose = pose.view(-1, 6)

            if len(result) == 0:
                result = np.copy(pose.cpu().detach().numpy())
            else:
                result = np.concatenate((result, pose.cpu().detach().numpy()), axis=0)

            trans_pose = compute_trans_pose(poses[j].cpu().data.numpy().astype(np.float64),
                                            poses[j + 1].cpu().data.numpy().astype(np.float64))

            if torch.cuda.is_available():
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1]).cuda()
            else:
                pose_truth = torch.FloatTensor(trans_pose[:, :, -1])

            rot_mat = torch.FloatTensor(trans_pose[:, :, :3]).cuda()

            euler = mat2euler(rot_mat)

            euler_loss += F.mse_loss(euler, pose[:, 3:])

            pose_loss += F.mse_loss(pose_truth, pose[:, :3])

            if len(truth_pose) == 0:
                truth_pose = np.copy(pose_truth.cpu().detach().numpy())
            else:
                truth_pose = np.concatenate((truth_pose, pose_truth.cpu().detach().numpy()), axis=0)

            if len(truth_euler) == 0:
                truth_euler = np.copy(euler.cpu().detach().numpy())
            else:
                truth_euler = np.concatenate((truth_euler, euler.cpu().detach().numpy()), axis=0)

        euler_loss /= (len(imgs) - 1)
        pose_loss /= (len(imgs) - 1)

        loss = pose_loss + euler_loss * 100

        aver_pose_loss += pose_loss.item()
        aver_loss += loss.item()

        aver_euler_loss += euler_loss.item()

        aver_n += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test Seq{}: Epoch [{}/{}] Step [{}/{}]: Loss {:.5} '
              'Pose {:.5} Euler {:.5}'.
              format(k, epoch + 1, args.epochs, i + 1, len(test_loader), loss.item(), pose_loss.item(),
                     euler_loss.item()))

        file_name = r'D:\SLAM\results\lvio\hard-100\result_seq' + str(k) + '_' + str(epoch) + '.csv'
        np.savetxt(file_name, result, delimiter=',')

        file_name = r'D:\SLAM\results\lvio\hard-100\truth_pose_seq' + str(k) + '_' + str(epoch) + '.csv'
        np.savetxt(file_name, truth_pose, delimiter=',')

        file_name = r'D:\SLAM\results\lvio\hard-100\truth_euler_seq' + str(k) + '_' + str(epoch) + '.csv'
        np.savetxt(file_name, truth_euler, delimiter=',')

    aver_loss /= aver_n
    aver_pose_loss /= aver_n
    aver_euler_loss /= aver_n
    print('Test Average: {}, Average_Loss {:.5} Pose_loss {:.5} Euler_loss {:.5}'
          .format(epoch + 1, aver_loss, aver_pose_loss, aver_euler_loss))

    return

def compute_trans_pose(ref_pose, tgt_pose):

    tmp_pose = np.copy(tgt_pose)
    tmp_pose[:, :, -1] -= ref_pose[:, :, -1]
    trans_pose = np.linalg.inv(ref_pose[:, :, :3]) @ tmp_pose
    return trans_pose

    # trans_pose = np.copy(tgt_pose)
    # trans_pose[:, :, -1] -= ref_pose[:, :, -1]

if __name__ == '__main__':
    main()
