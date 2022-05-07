import argparse

from utils.GenerateData import CropNMOMain

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hade')
    parser.add_argument('--CropSize', type=str, default='256,256')
    opt = parser.parse_args()

    CropSize = list(map(int, opt.CropSize.split(',')))
    CropNMOMain(opt.dataset, CropSize, SourcePath=r'E:\Spectrum', SaveRoot=r'P:\\Spectrum\\CropNMOData')
