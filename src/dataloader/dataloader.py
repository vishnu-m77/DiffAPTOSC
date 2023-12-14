# import pickle
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import src.dataloader.transforms as tr
import logging

"""
Currently using transforms defined by authors, but we can later on
replace functions by using torchvision transforms instead
"""
logging.getLogger('PIL').setLevel(logging.WARNING)


class APTOSDataset(Dataset):
    def __init__(self, data_path, total_image_num, type='train'):
        self.trainsize = (224, 224)
        self.total_image_num = total_image_num
        self.type = type
        self.data_path = data_path

        with open(data_path, "rb") as f:
            tr_dl = json.load(f)
        self.dataset = tr_dl

        # print(self.size)
        if self.type == 'train':
            logging.info("Initialize DataLoader for train dataset")
            self.dataset = random_image_selection(
                self.dataset, 0.7, total_image_num)
            self.size = len(self.dataset)
            self.transform_center = transforms.Compose([
                tr.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            logging.info("Initialize DataLoader for test dataset")
            if self.type == 'val':
                self.dataset = random_image_selection(
                    self.dataset, 0.1, total_image_num)
            elif self.type == 'test':
                self.dataset = random_image_selection(
                    self.dataset, 0.2, total_image_num)

            self.size = len(self.dataset)
            self.transform_center = transforms.Compose([
                tr.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        data_pac = self.dataset[index]
        img_path = data_pac['img_root']
        img = Image.open(img_path).convert('RGB')
        img_torch = self.transform_center(img)
        label = int(data_pac['label'])
        return img_torch, label

    def __len__(self):
        return self.size


class DataProcessor():
    def __init__(self, params, database_name="APTOS"):
        super(DataProcessor, self).__init__()
        self.database_name = database_name
        self.train_path = params["train_path"]
        self.test_path = params["test_path"]
        self.train_batch_size = params["train_batch_size"]
        self.test_batch_size = params["test_batch_size"]
        self.valid_batch_size = params["valid_batch_size"]
        if params["num_images"] > 100:
            self.total_image_num = params["num_images"]
        else:
            self.total_image_num = 100

    def get_dataloaders(self):
        train_data = APTOSDataset(
            data_path=self.train_path, total_image_num=self.total_image_num, type='train')
        test_data = APTOSDataset(
            data_path=self.test_path, total_image_num=self.total_image_num, type='test')
        val_data = APTOSDataset(
            data_path=self.test_path, total_image_num=self.total_image_num, type='val')
        train_loader = DataLoader(
            train_data,
            batch_size=self.train_batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_data,
            batch_size=self.test_batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            val_data,
            batch_size=self.valid_batch_size,
            shuffle=True
        )

        return train_loader, test_loader, valid_loader

def random_image_selection(original, total_image_num, multiplier):
    '''
    original -> contains the list of all the image locations and the labels from either test or train json
    multiplier -> 0.7 for training set and 0.3 for test set
    '''
    new_random = random.sample(original, int(multiplier*total_image_num))
    return new_random
