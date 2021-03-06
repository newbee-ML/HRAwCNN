# Reproduced code of HRAwCNN for automatic velocity analysis
This repository replicates a method of velocity analysis, but we do not provide the dataset because it is classified.
Cite: Ferreira R S, Oliveira D A B, Semin D G, et al. Automatic velocity analysis using a hybrid regression approach with convolutional neural networks[J]. IEEE Transactions on Geoscience and Remote Sensing, 2020, 59(5): 4464-4470.

## Data preparation
You need prepara two segy(sgy) files which includes velocity spectra and CMP gather infomation, and a label file which includes the velocity labels. You have to build the h5 file for the index of samples, as shown in https://github.com/newbee-ML/MIFN-Velocity-Picking/blob/master/utils/BuiltStkDataSet.py

## Implement
There are three parts for implementing the method proposed by Ferreira et al.: 
1) Generate the CropNMO dataset for training Xception Network. 
2) Training Xception Network. 
3) Predict processing

Tips:
You have to change a few path settings, if you want to test these method on your datasets.

### Generate the CropNMO dataset for training Xception Network
```cmd
python GenerateCNNData.py --dataset hade --CropSize 256,256
python GenerateCNNData.py --dataset dq8 --CropSize 256,256
```

### Training Xception Network
```cmd
python XceptionTrainMain.py
```
Training details:
- Velocity range [1000, 7000]
- NMO correction interval 50m/s, range [-1000, 1000]
- The shape of input NMO image is (256, 256)


### Predict processing

```cmd
python HRAwCNNPredMain.py
```
Prediction details:
- The shape of input NMO image is (256, 256)
- Time stride is 100 pixel


## Test Results on two field datasets


| **DataSet** 	| **lrStart** 	| **optimizer** 	| **trainBS** 	|   **VMAE**   	|
|:-----------:	|:-----------:	|:-------------:	|:-----------:	|:------------:	|
|   **hade**  	|  **0.0001** 	|    **adam**   	|    **32**   	| **173.5236** 	|
|   **hade**  	|    0.0001   	|      adam     	|      16     	|   190.6544   	|
|   **hade**  	|    0.001    	|      adam     	|      32     	|   191.1893   	|
|   **hade**  	|    0.001    	|      adam     	|      16     	|   225.2926   	|
|   **dq8**   	|  **0.0001** 	|    **adam**   	|    **32**   	| **778.9616** 	|
|   **dq8**   	|    0.001    	|      adam     	|      32     	|   823.6368   	|
|   **dq8**   	|    0.0001   	|      adam     	|      16     	|   887.0123   	|
|   **dq8**   	|    0.001    	|      adam     	|      16     	|   1042.129   	|