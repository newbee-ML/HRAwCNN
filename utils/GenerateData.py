"""
generate the NMO dataset for training Xception Network
---
Author: Wang Hongtao | stolzpi@163.com
Date: 2022-05-05
"""

import os

from tqdm import tqdm
import numpy as np
import segyio
import h5py
import torch.utils.data as data

from utils.LoadData import interpolation, LoadSource

"""
Load source data
"""


class GthLoad(data.Dataset):
    def __init__(self, SegyDict, H5Dict, LabelDict, IndexList):
        self.SegyDict = SegyDict
        self.H5Dict = H5Dict
        self.LabelDict = LabelDict
        self.IndexList = IndexList
        self.tInt = np.array(SegyDict['gth'].samples)

    def __getitem__(self, ind):
        #########################################
        # load the data from segy and h5 file
        #########################################
        index = self.IndexList[ind]
        GthIndex = np.array(self.H5Dict['gth'][index]['GatherIndex'])
        line, cdp = index.split('_')
        Gather = np.array(self.SegyDict['gth'].trace.raw[GthIndex[0]: GthIndex[1]].T)
        OInd = np.array(self.SegyDict['gth'].attributes(segyio.TraceField.offset)[GthIndex[0]: GthIndex[1]])
        try:
            VelLab = np.array(self.LabelDict[int(line)][int(cdp)])
        except KeyError:
            VelLab = np.array(self.LabelDict[line][cdp])
        #########################################
        # interpolate the velocity curve
        #########################################
        InterpVel = interpolation(VelLab, self.tInt)
        return Gather, OInd, InterpVel, index

    def __len__(self):
        return len(self.IndexList)



def LocalNMO(Gather, vel, CropRange, tInt, OffsetInt, OutWid):
    LocaltInt = tInt[CropRange[0]:CropRange[1]]/1000
    XTime = (OffsetInt/vel)**2
    TravelT = np.ones((len(LocaltInt), len(OffsetInt)))
    TravelT = (LocaltInt * TravelT.T).T * XTime
    TravelT = (np.sqrt(TravelT)*1000).astype(np.int32)
    TravelT[TravelT<0] = 0
    TravelT[TravelT>=Gather.shape[0]] = Gather.shape[0]-1
    NMOResult = np.zeros((len(LocaltInt), OutWid))
    IterWid = np.min([len(OffsetInt), OutWid])
    for j in range(IterWid):
       NMOResult[:, j] = Gather[TravelT[:, j], j]
    return NMOResult


def CropNMOMain(dataset, CropWin=(120, 120), SourcePath=r'E:\Spectrum', SaveRoot=r'P:\Spectrum\HRAwCNN\data'):
    DataSetPath = os.path.join(SourcePath, dataset)
    SegyDict, H5Dict, LabelDict, IndexDict = LoadSource(DataSetPath)
    tInt = np.array(SegyDict['gth'].samples)
    # get index for these three datasets
    for name, IndexList in IndexDict.items():
        DataIter = GthLoad(SegyDict, H5Dict, LabelDict, IndexList)
        path = os.path.join(SaveRoot, dataset)
        if not os.path.exists(path): os.makedirs(path)
        H5file = h5py.File(os.path.join(path, '%s.h5'%name), 'w')
        bar = tqdm(total=len(DataIter))
        for Gather, OInd, InterpVel, index in DataIter:
            VelInt = np.arange(-1000, 1001, 50)
            CropIndex = np.arange(0, Gather.shape[0], CropWin[0])
            for num in range(len(CropIndex)-1):
                RowS, RowE = CropIndex[num: (num+2)]
                for DiffVel in VelInt:
                    RefVel = InterpVel[int((RowS+RowE-1)/2), 1] + DiffVel
                    if RefVel < 0: continue
                    CropNumGth = LocalNMO(Gather, RefVel, [RowS, RowE], tInt, OInd, CropWin[1])
                    CroptInt = tInt[RowS: RowE]
                    sample = H5file.create_group('%s_%d_%d' % (index, num, DiffVel))
                    sample.create_dataset('gather', data=CropNumGth.astype(np.float32))
                    sample.create_dataset('RefVel', data=RefVel)
                    sample.create_dataset('tInt', data=CroptInt.astype(np.float32))
            bar.update(1)
            bar.set_description('%s_%s'%(dataset, index))
        H5file.close()
        bar.close()

