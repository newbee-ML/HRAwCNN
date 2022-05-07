import os

import numpy as np
import segyio
import h5py
import torch
import torch.utils.data as data
from scipy import interpolate
from torchvision import transforms


"""
Loading Sample Data from segy, h5, npy file
"""


# make ground truth curve
def interpolation(label_point, t_interval, v_interval=None):
    # sort the label points
    label_point = np.array(sorted(label_point, key=lambda t_v: t_v[0]))

    # ensure the input is int
    t0_vec = np.array(t_interval).astype(int)

    # get the ground truth curve using interpolation
    peaks_selected = np.array(label_point)
    func = interpolate.interp1d(peaks_selected[:, 0], peaks_selected[:, 1], kind='linear', fill_value="extrapolate")
    y = func(t0_vec)
    if v_interval is not None:
        v_vec = np.array(v_interval).astype(int) 
        y = np.clip(y, v_vec[0], v_vec[-1])

    return np.hstack((t0_vec.reshape((-1, 1)), y.reshape((-1, 1))))


def ScaleImage(SpecImg, resize_n, device='cpu'):
    _, H, W = SpecImg.shape
    SpecImg = torch.tensor(SpecImg, device=device).view(1, 2, H, W)
    transform = transforms.Resize(resize_n).cuda(device)
    AugImg = transform(SpecImg)  
    return AugImg.squeeze()


def LoadSource(DataSetPath):
    # load segy data
    SegyName = {'pwr': 'vel.pwr.sgy',
                'gth': 'vel.gth.sgy'}
    SegyDict = {}
    for name, path in SegyName.items():
        SegyDict.setdefault(name, segyio.open(os.path.join(DataSetPath, 'segy', path), "r", strict=False))

    # load h5 file
    H5Name = {'pwr': 'SpecInfo.h5',
              'gth': 'GatherInfo.h5'}
    H5Dict = {}
    for name, path in H5Name.items():
        H5Dict.setdefault(name, h5py.File(os.path.join(DataSetPath, 'h5File', path), 'r'))

    # load label.npy
    LabelDict = np.load(os.path.join(DataSetPath, 't_v_labels.npy'), allow_pickle=True).item()
    HaveLabelIndex = []
    for lineN in LabelDict.keys():
        for cdpN in LabelDict[lineN].keys():
            HaveLabelIndex.append('%s_%s' % (lineN, cdpN))
    pwr_index = set(H5Dict['pwr'].keys())
    gth_index = set(H5Dict['gth'].keys())
    Index = sorted(list(pwr_index & (gth_index & set(HaveLabelIndex))))
    Index = sorted(list((pwr_index) & (gth_index & set(HaveLabelIndex))))
    IndexDict = {}
    for index in Index:
        line, cdp = index.split('_')
        IndexDict.setdefault(int(line), [])
        IndexDict[int(line)].append(int(cdp))
    LineIndex = sorted(list(IndexDict.keys()))
    # use the last 20% for test set
    LastSplit1, LastSplit2 = int(len(LineIndex)*0.6), int(len(LineIndex)*0.8)
    # use the first sr% (seed rate) for train set and the other for valid set
    MedSplit = int(LastSplit1)
    trainLine, validLine, testLine = LineIndex[:MedSplit], LineIndex[LastSplit1: LastSplit2], LineIndex[LastSplit2:]
    trainIndex, validIndex, testIndex = [], [], []
    for line in trainLine:
        for cdp in IndexDict[line]:
            trainIndex.append('%d_%d' % (line, cdp))
    for line in validLine:
        for cdp in IndexDict[line]:
            validIndex.append('%d_%d' % (line, cdp))
    for line in testLine:
        for cdp in IndexDict[line]:
            testIndex.append('%d_%d' % (line, cdp))
    IndexDict = {'train': trainIndex, 'valid': validIndex, 'test': testIndex}
    return SegyDict, H5Dict, LabelDict, IndexDict


# --------- load gather & spectrum data fron segy, h5 and label.npy ------------
def LoadSingleData(SegyDict, H5Dict, LabelDict, index, mode='train'):
    # data dict
    DataDict = {}
    GthIndex = np.array(H5Dict['gth'][index]['GatherIndex'])
    PwrIndex = np.array(H5Dict['pwr'][index]['SpecIndex'])
    line, cdp = index.split('_')

    DataDict.setdefault('gather', np.array(SegyDict['gth'].trace.raw[GthIndex[0]: GthIndex[1]].T))
    DataDict.setdefault('oInt', np.array(SegyDict['gth'].attributes(segyio.TraceField.offset)[GthIndex[0]: GthIndex[1]]))
    DataDict.setdefault('spectrum', np.array(SegyDict['pwr'].trace.raw[PwrIndex[0]: PwrIndex[1]].T))
    DataDict.setdefault('vInt', np.array(SegyDict['pwr'].attributes(segyio.TraceField.offset)[PwrIndex[0]: PwrIndex[1]]))

    if mode == 'train':
        try:
            DataDict.setdefault('label', np.array(LabelDict[int(line)][int(cdp)]))
        except KeyError:
            DataDict.setdefault('label', np.array(LabelDict[str(line)][str(cdp)]))

    return DataDict


# -------- make label vector --------------------------------------
def MakeLabel(value, VRange):
    LocIndex = np.where((VRange-value)>0)[0][0]-1
    return LocIndex


# ------- make feature map comb -----------------------
def FeatConcat(gather, RefVel, tInt, device):
    tIntMap = (np.ones_like(gather).T * tInt).T
    gather = torch.tensor(gather, device=device, dtype=torch.float32)
    gather = gather / torch.max(torch.abs(gather))
    RefVelMap = torch.ones_like(gather, device=device, dtype=torch.float32) * (RefVel/10000)
    tIntMap = torch.tensor(tIntMap/10000, device=device, dtype=torch.float32)
    Feats = torch.concat((gather.unsqueeze(0), RefVelMap.unsqueeze(0), tIntMap.unsqueeze(0)), dim=0)
    return Feats


# -------- dataloader for training Xception Network ---------------
"""
Load Data: 
    Feats: feature maps       | shape = 3 * 120 * 120
    label: vector label       | float
    self.index[index]: sample index    | string
"""
class CropNMOLoad(data.Dataset):
    def __init__(self, RootPath, mode='train', device='cpu'):
        self.H5file = h5py.File(os.path.join(RootPath, '%s.h5' % mode), 'r')
        self.IndexList = list(self.H5file.keys())
        self.device = device

    def __getitem__(self, ind):
        index = self.IndexList[ind]
        SampleData = self.H5file[index]
        gather = np.array(SampleData['gather'])
        RefVel = np.array(SampleData['RefVel']).item()
        tInt = np.array(SampleData['tInt'])
        DiffVel = int(index.split('_')[-1])
        Feats = FeatConcat(gather, RefVel, tInt, self.device)
        return Feats, DiffVel, index

    def __len__(self):
        return len(self.IndexList)


# -------- load all source data for all stage predict -----------
class PredLoad(data.Dataset):
    def __init__(self, SegyDict, H5Dict, LabelDict, index):
        self.SegyDict = SegyDict
        self.LabelDict = LabelDict
        self.H5Dict = H5Dict
        self.IndexList = index
        self.tInt = np.array(SegyDict['gth'].samples)
        self.t0Int = np.array(SegyDict['pwr'].samples)

    def __getitem__(self, ind):
        # load the data from segy and h5 file
        index = self.IndexList[ind]
        DataDict = LoadSingleData(self.SegyDict, self.H5Dict, self.LabelDict,
                                  index)
        PwrImg = DataDict['spectrum']
        Gather = DataDict['gather']
        # interpolate velocity curve
        VCAllRes = interpolation(DataDict['label'], self.t0Int, v_interval=DataDict['vInt'])
        # summary sample axis info
        InfoDict = {'vInt': DataDict['vInt'], 'oInt': DataDict['oInt']}
        return PwrImg, Gather, VCAllRes, InfoDict, index

    def __len__(self):
        return len(self.IndexList)


