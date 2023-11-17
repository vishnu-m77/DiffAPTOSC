# DiffMIC
Final Project for AMATH 495
## Datasets

1. Download [HAM10000](https://challenge.isic-archive.com/data/#2018) or [APTOS2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) dataset. Your dataset folder under "your_data_path" should be like:

dataset/isic2018/

     images/...
     
     ISIC2018_Task3_Training_GroundTruth.csv
     
     isic2018_train.pkl

     isic2018_test.pkl

dataset/aptos/

     train/...
     
     train.csv
     
     aptos_train.pkl

     aptos_test.pkl

.pkl file contains the list of data whose element is a dictionary with the format as ``{'img_root':image_path,'label':label}``

## DCG

