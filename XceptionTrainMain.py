from XceptionTrainFunc import GetTrainPara, train
from Tuning.tuning import ListPara, ParaStr2Dict, UpdateOpt
import os

#################################
# experiment setting
#################################
# Ep 1-16
ParaDict = {'DataSet': ['str', ['hade', 'dq8']], 
            'OutputPath': ['str', ['F:\\VelocityPicking\\HRAwCNN\\Xception']], 
            'SeedRate': ['float', [1.0]], 
            'trainBS': ['int', [16, 32]],
            'StopMax': ['int', [50]], 
            'lrStart': ['float', [0.001, 0.0001]],
            'optimizer': ['str', ['adam', 'sgd']]}

start = 10

#################################
# training
#################################
# get the experiment (ep) list
EpList = ListPara(ParaDict)
# get default training parameters
OptDefault = GetTrainPara()
for ind, EpName in enumerate(EpList):
    # try:
    # get the ep para dict
    EpDict = ParaStr2Dict(EpName, ParaDict)
    EpDict.setdefault('EpName', 'Ep-%d' % (ind+start))
    # judge whether done before
    if os.path.exists(os.path.join(EpDict['OutputPath'], 'Ep-%d' % (ind+start), 'Result.csv')):
        continue
    if os.path.exists(os.path.join(EpDict['OutputPath'], 'Ep-%d' % (ind+start), 'model', 'Best.pth')):
        EpDict.setdefault('ReTrain', 0)
    else:
        EpDict.setdefault('ReTrain', 1)
    # update the para
    EpOpt = UpdateOpt(EpDict, OptDefault)
    # start this experiment
    train(EpOpt)
    # except:
    #     print(EpName)
    #     continue
