# DiffAPTOSC

Final Project for AMATH 495

The project is based on the paper [DiffMIC: Dual-Guidance Diffusion Network for Medical Image Classification](https://arxiv.org/abs/2303.10610).

`main.py` calls the modules in `src/`. The program loads a saved DCG checkpoint `saved_dcg.pth` if it exists. It trains and saves a DCG checkpoint otherwise. Similarly, for diffusion, the program loads a saved diffusion checkpoint `saved_diff.pth` if it exists. It trains and saves a diffusion checkpoint otherwise. After loading the DCG and diffusion models, the program runs inference on the test images and outputs the predicted classification [Classes 1 - 5].

## Dataset

Download [APTOS2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) dataset. Your dataset folder under "your_data_path" should be like:

dataset/aptos/

     test/...
     
     train/...

     aptos_test.json
     
     apts_train.json

## Parameters

NOTE: If changes are made to "data":"num_classes" , "diffusion":"timesteps" params; make sure to make those changes in "unet" params.

## dataloader

## DCG: Dual-granularity Conditional Guidance

`main.py`: class DCG is defined along with functions that train the DCG model, and to load the DCG model from a saved checkpoint.

`networks.py`: The networks used in DCG are defined: Saliency Map, Global Network, Downsample Network, Local Network, and Attention Module.

`utils.py`: Helper functions used in DCG are defined.

## Diffusion

## UNet Model

## Metrics

## Plots

## Report

## Thanks
Code is largely based on [scott-yjyang/DiffMIC](https://github.com/scott-yjyang/DiffMIC), [XzwHan/CARD](https://github.com/XzwHan/CARD), [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), [MedSegDiff](https://github.com/WuJunde/MedSegDiff/tree/master), [nyukat/GMIC](https://github.com/nyukat/GMIC)

