# DMA
This repo is the official implementation for:\
[Pattern Recognition] [Selective Intra-and Inter-Slice Interaction for Efficient Anisotropic Medical Image Segmentation](https://www.sciencedirect.com/science/article/pii/S0031320325015584).\
(The details of our ACSFormer can be found at the models directory in this repo or in the paper.)

## Requirements
* python 3.6
* pytorch 1.8.0
* torchvision 0.9.0
* more details please see the requirements.txt

## Datasets
* The INSTANCE dataset could be acquired from [here](https://instance.grand-challenge.org/).
* The PROSTATE dataset could be acquired from [here](http://medicaldecathlon.com/).
* The MosMed dataset could be acquired from [here](https://www.kaggle.com/datasets/mathurinache/mosmeddata-chest-ct-scans-with-covid19).
* The ACDC dataset could be acquired from [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/). 

## Training
Commands for training
```
python train2p5D_INSTANCE.py
python train2p5D_Prostate.py
```
## Testing
Commands for testing
``` 
python test2p5D.py
python test2p5D_Prostate.py
```
