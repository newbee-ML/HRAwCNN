

"""
The main file for train VGG16 for velocity picking
Author: Hongtao Wang | stolzpi@163.com
"""
import sys
from ast import Raise

sys.path.append('..')
import argparse
import copy
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model.Xception import xception
from utils.LoadData import CropNMOLoad
from utils.logger import MyLog
warnings.filterwarnings("ignore")


"""
Initialize the folder
"""

def CheckSavePath(opt, BaseName):
    basicFile = ['log', 'model', 'TBLog']
    SavePath = os.path.join(opt.OutputPath, BaseName)
    if opt.ReTrain:
        if os.path.exists(SavePath):
            shutil.rmtree(SavePath)
        if not os.path.exists(SavePath):
            for file in basicFile:
                Path = os.path.join(SavePath, file)
                os.makedirs(Path)

"""
Save the training parameters
"""
def SaveParameters(opt, BaseName):
    ParaDict = opt.__dict__
    ParaDict = {key: [value] for key, value in ParaDict.items()}
    ParaDF = pd.DataFrame(ParaDict)
    ParaDF.to_csv(os.path.join(opt.OutputPath, BaseName, 'TrainPara.csv'))


"""
Get the hyper parameters
"""
def GetTrainPara():
    parser = argparse.ArgumentParser()
    ###########################################################################
    # path setting
    ###########################################################################
    parser.add_argument('--EpName', type=str, default='Ep-1', help='The index of the experiment')
    parser.add_argument('--DataSetRoot', type=str, default='P:\\Spectrum\\CropNMOData')
    parser.add_argument('--OutputPath', type=str, help='Path of Output')
    parser.add_argument('--ReTrain', type=int, default=1)
    ###########################################################################
    # load data setting
    ###########################################################################
    parser.add_argument('--DataSet', type=str, default='dq8', help='Dataset List')
    parser.add_argument('--SeedRate', type=float, default=1)
    ###########################################################################
    # training setting
    ###########################################################################
    parser.add_argument('--StopMax', type=int, default=10)
    parser.add_argument('--GPUNO', type=int, default=0)
    parser.add_argument('--MaxIter', type=int, default=20000, help='max iteration')
    parser.add_argument('--SaveIter', type=int, default=50, help='checkpoint each SaveIter')
    parser.add_argument('--MsgIter', type=int, default=10, help='log the loss each MsgIter')
    parser.add_argument('--lrStart', type=float, default=0.01, help='the beginning learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help=r"the optimizer of training, 'adam' or 'sgd'")
    parser.add_argument('--PretrainModel', type=str, help='The path of pretrain model to train (Path)')
    parser.add_argument('--trainBS', type=int, default=32, help='The batchsize of train')
    parser.add_argument('--valBS', type=int, default=32, help='The batchsize of valid')
    parser.add_argument('--valNum', type=int, default=10000, help='The batchsize of valid')
    opt = parser.parse_args()
    return opt
    

"""
Main train function
"""

def train(opt):
    ####################
    # base setting
    ####################
    BaseName = opt.EpName
    # check output folder and check path
    CheckSavePath(opt, BaseName)
    DataSetPath = os.path.join(opt.DataSetRoot, opt.DataSet)
    TBPath = os.path.join(opt.OutputPath, BaseName, 'TBLog')
    writer = SummaryWriter(TBPath)
    BestPath = os.path.join(opt.OutputPath, BaseName, 'model', 'Best.pth')
    LogPath = os.path.join(opt.OutputPath, BaseName, 'log')
    logger = MyLog(BaseName, LogPath)
    logger.info('%s start to train ...' % BaseName)
    # save the train parameters to csv
    SaveParameters(opt, BaseName)

    ##################################
    # build the data loader
    ##################################
    # check gpu is available
    if torch.cuda.device_count() > 0:
        device = opt.GPUNO
    else:
        device = 'cpu'
    # build data loader
    ds = CropNMOLoad(DataSetPath, mode='train', device=device)
    dsval = CropNMOLoad(DataSetPath, mode='valid', device=device)
    dl = DataLoader(ds,
                    batch_size=opt.trainBS,
                    shuffle=True,
                    pin_memory=False,
                    num_workers=0,
                    drop_last=True)
    dlval = DataLoader(dsval,
                       batch_size=opt.valBS,
                       shuffle=True,
                       pin_memory=False,
                       num_workers=0,
                       drop_last=True)

    ###################################
    # load the network
    ###################################
    # load network
    net = xception(pretrained=True, num_classes=1)

    if device is not 'cpu':
        net = net.cuda(device)
    net.train()

    # load pretrain model or last model
    if opt.PretrainModel is None:
        if os.path.exists(BestPath):
            print("Load Last Model Successfully!")
            LoadModelDict = torch.load(BestPath)
            net.load_state_dict(LoadModelDict['Weights'])
            TrainParaDict = LoadModelDict['TrainParas']
            countIter, epoch = TrainParaDict['it'], TrainParaDict['epoch']
            BestValidLoss, lrStart = TrainParaDict['bestLoss'], TrainParaDict['lr']
        else:
            print("Start a new training!")
            countIter, epoch, lrStart, BestValidLoss = 0, 1, opt.lrStart, 1e10
    else:
        print("Load Pretrain Model Successfully!")
        LoadModelDict = torch.load(opt.PretrainModel)
        net.load_state_dict(LoadModelDict['Weights'])
        countIter, epoch, lrStart, BestValidLoss = 0, 1, opt.lrStart, 1e10
    
    # loss setting
    criterion = nn.MSELoss(reduction='mean')
    # define the optimizer
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lrStart)
    elif opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lrStart, momentum=0.9)
    else:
        Raise("Error: invalid optimizer") 

    # define the lr_scheduler of the optimizer
    scheduler = MultiStepLR(optimizer, [10], 0.1)

    ####################################
    # training iteration 
    ####################################

    # initialize
    LossList, EarlyStopCount = [], 0

    # start the iteration
    diter = iter(dl)
    for _ in range(opt.MaxIter):
        if countIter % len(dl) == 0 and countIter > 0:
            epoch += 1
            scheduler.step()
        countIter += 1
        try:
            Feats, DiffVel, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            Feats, DiffVel, _ = next(diter)
        optimizer.zero_grad()
        out = net(Feats)
        # print(out[0], DiffVel[0])
        # PLotMultiTensor(Feats[0])
        
        out = out.squeeze()
        # compute loss
        loss = criterion(out, DiffVel.cuda(device).float())
        # update parameters
        loss.backward()
        optimizer.step()
        LossList.append(loss.item())
        # save loss lr & seg map
        writer.add_scalar('Train-Loss', loss.item(), global_step=countIter)
        writer.add_scalar('Train-Lr', optimizer.param_groups[0]['lr'], global_step=countIter)

        # print the log per opt.MsgIter
        if countIter % opt.MsgIter == 0:
            lr = optimizer.param_groups[0]['lr']
            msg = 'it: %d/%d, epoch: %d, lr: %.6f, train-loss: %.7f' % (countIter, opt.MaxIter, epoch, lr, sum(LossList) / len(LossList))
            logger.info(msg)
            LossList = []

        
        # check points
        if countIter % opt.SaveIter == 0:  
            net.eval()
            # evaluator
            with torch.no_grad():
                ValidLoss, ValidMAE = EvaluateValid(net, dlval, criterion)
                writer.add_scalar('Valid-Loss', ValidLoss, global_step=countIter)
                writer.add_scalar('Valid-MAE', ValidMAE, global_step=countIter)

            if ValidLoss < BestValidLoss:
                BestValidLoss = copy.deepcopy(ValidLoss)
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                StateDict = {
                    'TrainParas': {'lr': optimizer.param_groups[0]['lr'], 
                                   'it': countIter,
                                   'epoch': epoch,
                                   'bestLoss': BestValidLoss},
                    'Weights': state}
                torch.save(StateDict, BestPath)
                EarlyStopCount = 0
            else:
                # count 1 time
                EarlyStopCount += 1
                # reload checkpoint pth
                if os.path.exists(BestPath):
                    net.load_state_dict(torch.load(BestPath)['Weights'])
                # if do not decreate for 10 times then early stop
                if EarlyStopCount > opt.StopMax:
                    break
            
            # write the valid log
            try:
                logger.info('it: %d/%d, epoch: %d, BestLoss: %.6f, Loss: %.6f, MAE: %.3f' % (countIter, opt.MaxIter, epoch, BestValidLoss, ValidLoss, ValidMAE))
            except TypeError:
                logger.info('it: %d/%d, epoch: %d, TypeError')
            net.train()

    # save the finish csv
    ResultDF = pd.DataFrame({'BestValidLoss': [BestValidLoss]})
    ResultDF.to_csv(os.path.join(opt.OutputPath, BaseName, 'Result.csv'))


# main function for valid processing
def EvaluateValid(net, DataLoader, criterion, ValidNum=2048):
    """
    main function for valid processing in trainSingle.py

    Params:
    - net: the network, type=class
    - DataLoader: data loader, type=class
    - criterion: loss function
    ---

    Return:
    - mean valid loss
    """
    # init save list and path
    LossAvg, MAE, count = [], [], 0
    # valid iteration
    for _, (FuseImg, label, _) in enumerate(DataLoader):
        out = net(FuseImg)
        label = label.cuda(FuseImg.device)
        # compute loss
        loss = criterion(out.squeeze(), label.squeeze())
        LossAvg.append(loss.item())
        count += out.shape[0]
        MAE.append(torch.mean(torch.abs(out.squeeze()-label.squeeze())).item())
        if count > ValidNum: break

    return sum(LossAvg)/len(LossAvg), sum(MAE)/len(MAE)

