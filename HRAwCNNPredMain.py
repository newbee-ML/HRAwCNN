from HRAwCNNPredFunc import GetPredPara, PredFunc
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    for i in range(1, 9):
        # get test parameters
        OptDefault = GetPredPara()
        OptDefault.EpName = i
        # start this experiment
        PredFunc(OptDefault)
