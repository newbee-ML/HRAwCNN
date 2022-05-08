import sys

sys.path.append('..')
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.LoadData import LoadSource, PredLoad

matplotlib.use('Agg')
import warnings

from model.Xception import xception
from model.InitialVelocity import InitialVelocity
from utils.LoadData import LoadSource, interpolation, FeatConcat
from utils.PlotTools import plot_spectrum, PLotMultiTensor
from utils.GenerateData import LocalNMO
warnings.filterwarnings("ignore")


def GetPredPara():
    parser = argparse.ArgumentParser()
    parser.add_argument('--OutputPath', type=str, default='F:\\VelocityPicking\\HRAwCNN\\Xception', help='Root Path of Output Folder')
    parser.add_argument('--DataSetRoot', type=str, default='E:\\Spectrum', help='Root Path of Source Data')
    parser.add_argument('--CropSize', type=str, default='120,250')
    parser.add_argument('--EpName', type=int, help='Model Path')
    parser.add_argument('--stride', type=int, default=100, help='The Stride of prediction')
    parser.add_argument('--Resave', type=int, default=0)
    parser.add_argument('--GPUNO', type=int, default=0, help='Used GPU Index')
    parser.add_argument('--VisualNum', type=int, default=16, help='The number of validation samples')

    opt = parser.parse_args()
    return opt


def InvertVel(output, t0Ind, vInd, RowSplit):
    VCenter = (vInd[:-1] + vInd[1:])/2
    MaxIndex = np.argmax(output, axis=1)
    VEst = VCenter[MaxIndex]
    RowSplit[RowSplit==len(t0Ind)] = -1
    T0Est = (t0Ind[RowSplit[:, 0]] + t0Ind[RowSplit[:, 1]])/2
    AutoCurve = np.array([T0Est, VEst]).T
    return AutoCurve


def PredFunc(opt):
    ####################
    # base setting
    ####################
    BasePath = os.path.join(opt.OutputPath, 'Ep-%d'%opt.EpName)
    # setting model parameters
    ParaDict = pd.read_csv(os.path.join(BasePath, 'TrainPara.csv')).to_dict()
    PredSet = str(ParaDict['DataSet'][0])
    DataSetPath = os.path.join(opt.DataSetRoot, PredSet)
    
    # check output folder
    OutputPath = os.path.join(BasePath, 'predict', PredSet)
    if not os.path.exists(OutputPath): os.makedirs(OutputPath) 
    PlotRoot = os.path.join(OutputPath, 'fig')
    if not os.path.exists(PlotRoot): os.makedirs(PlotRoot)
    CropSize = list(map(int, opt.CropSize.split(',')))
    ModelPath = os.path.join(BasePath, 'model', 'Best.pth')
    PredictPath = os.path.join(OutputPath, '%s-VMAE.npy' % PredSet)
    # check gpu is available
    if torch.cuda.device_count() > 0:
        device = opt.GPUNO
    else:
        device = 'cpu'

    #######################################
    # load data from segy, csv index file
    #######################################
    # load source file
    SegyDict, H5Dict, LabelDict, IndexDict = LoadSource(DataSetPath)
    testIndex = IndexDict['test']
    print('Test Num %d' % len(testIndex))
    
    ds = PredLoad(SegyDict, H5Dict, LabelDict, testIndex)

    ###################################
    # load the Xception network
    ###################################
    # load network
    net = xception(num_classes=1)

    if device is not 'cpu':
        net = net.cuda(device)
    net.eval()
    # Load the weights of network
    if os.path.exists(ModelPath):
        print("Load Model Successfully! \n(%s)" % ModelPath)
        net.load_state_dict(torch.load(ModelPath)['Weights'])
    else:
        print("There is no such model file!")

    bar = tqdm(total=len(ds))
    VMAEList, CountVisual = [], 0
    # predict all of the dataset
    with torch.no_grad():
        for PwrImg, Gather, MPCurve, InfoDict, FMIndex in ds:
            # load basic axis infomation
            t0Int, tInt = ds.t0Int, ds.tInt
            vInt, oInt = InfoDict['vInt'], InfoDict['oInt']
            t0d, td = t0Int[1]-t0Int[0], tInt[1]-tInt[0]
            MaxtInt = int(len(t0Int)*t0d/td)
            # crop center index
            CroptIndex = np.arange(int(CropSize[0]/2+1), MaxtInt-int(CropSize[0]/2)-2, opt.stride)
            ################################
            # get initial velocity 
            ################################
            InitVel = InitialVelocity(PwrImg)
            InitVel = InitVel*(vInt[1]-vInt[0]) + vInt[0]
            Cropt0Index = (CroptIndex*td/t0d).astype(np.int32)
            CroptIndex = CroptIndex.astype(np.int32)
            CropCenterVel = {CroptIndex[ind]*td: [InitVel[Cropt0Index[ind]]] for ind in range(len(CroptIndex))}
            ################################
            # Tuned velocity picking
            ################################
            for ind, time in enumerate(list(CropCenterVel.keys())):
                # get crop index
                CropUp = CroptIndex[ind]-int(CropSize[0]/2)
                CropDown = CropUp + CropSize[0]
                CropRange = (CropUp, CropDown)
                count = 0
                while True:
                    VelIter = CropCenterVel[time][-1]
                    # get NMO velocity 
                    NMOResult = LocalNMO(Gather, VelIter, CropRange, tInt, oInt,CropSize[1])
                    InputMap = FeatConcat(NMOResult, VelIter, tInt[CropUp: CropDown], device)
                    
                    # predict
                    DeltaVel = net(InputMap.unsqueeze(0))
                    DeltaVel = DeltaVel.item()
                    NewVel = CropCenterVel[time][-1]+DeltaVel
                    NewVel = 100 if NewVel < 0 else NewVel
                    CropCenterVel[time].append(NewVel)
                    # print(DeltaVel)
                    # print(CropCenterVel[time])
                    # PLotMultiTensor(InputMap)
                    count += 1
                    if count > 0 or abs(DeltaVel) < 1:
                        break
            AutoCurve = [[time, VelList[-1]] for time, VelList in CropCenterVel.items()]
            APInterp = interpolation(AutoCurve, ds.t0Int)
            VMAEList.append([FMIndex, np.mean(np.abs(APInterp[:, 1]-MPCurve[:, 1]))])
            bar.update(1)
            bar.set_description('%s-%s: VMAE %.3f' % (PredSet, FMIndex, VMAEList[-1][-1]))
            if CountVisual < opt.VisualNum:
                plot_spectrum(PwrImg, t0Int, vInt, VelCurve=[APInterp, MPCurve], save_path=os.path.join(PlotRoot, '%s-result.png')%FMIndex)
                InitVelList = np.array([t0Int, InitVel]).T
                plot_spectrum(PwrImg, t0Int, vInt, VelCurve=[InitVelList, MPCurve], save_path=os.path.join(PlotRoot, '%s-InitVel.png')%FMIndex)
                CountVisual+=1
    bar.close()
    # save predict results
    np.save(PredictPath, np.array(VMAEList))
    print('%s Ep-%d VMAE = %.3f' % (PredSet, opt.EpName, np.mean(np.array(VMAEList)[:, 1].astype(np.float32))))



