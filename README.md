# DiffAPTOSC

Final Project for AMATH 495

The project is based on the paper [DiffMIC: Dual-Guidance Diffusion Network for Medical Image Classification](https://arxiv.org/abs/2303.10610).

Run the code: `python3 ./main.py`

`main.py` calls the modules in `src/`. The program loads a saved DCG checkpoint `saved_dcg.pth` if it exists. It trains and saves a DCG checkpoint otherwise. Similarly, for diffusion, the program loads a saved diffusion checkpoint `saved_diff.pth` if it exists. It trains and saves a diffusion checkpoint otherwise. After loading the DCG and diffusion models, the program runs inference on the test images and outputs the predicted classification [5 Classes: 0, 1, 2, 3, 4].

## Dataset

Download [APTOS2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) dataset. Your dataset folder under "your_data_path" should be like:

dataset/aptos/

     test/...

     train/...

     aptos_test.json

     apts_train.json

## Parameters

## Dataloader

`dataloader.py`: Using `DataProcessor.get_dataloaders()`, we get the train and test data, images and labels included. Train dataloader is used in DCG and Diffusion process for training. Test dataloader is used for evaluation of Diffusion.

`transforms.py`: Transformation functions used in Data Pre-processing.

## DCG: Dual-granularity Conditional Guidance

`main.py`: class DCG is defined along with functions that train the DCG model, and to load the DCG model from a saved checkpoint.

`networks.py`: The networks used in DCG are defined: Saliency Map, Global Network, Downsample Network, Local Network, and Attention Module.

`utils.py`: Helper functions used in DCG are defined.

## Diffusion

`diffusion.py`: Contains code for forward and reverse diffusion.

## UNet Model

`unet_model.py`: Contains the UNet model used in diffusion.

## Metrics

`metrics.py`: Contains the code for the classification metrics: Accuracy, f1 score, and t-SNE.

## Plots

Plots are generated in the `plots/` directory. `dcg_loss.png` is generated during the training of the DCG, and `diffusion_loss.png` is generated during the training of the diffusion model. `t-SNE` plots are generated for different timesteps `t1, t2, t3` in the diffusion parameters `params["diffusion"]["t-sne"]` during the inference step.

## Report

`project.log`: Logs during runtime are saved in this file.

`report.txt`: Accuracy of the DCG, and the diffusion model are saved in this file. The f1 score, and the t-SNE values are also saved.

## Thanks

Code is largely based on [scott-yjyang/DiffMIC](https://github.com/scott-yjyang/DiffMIC), [XzwHan/CARD](https://github.com/XzwHan/CARD), [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), [MedSegDiff](https://github.com/WuJunde/MedSegDiff/tree/master), [nyukat/GMIC](https://github.com/nyukat/GMIC)
