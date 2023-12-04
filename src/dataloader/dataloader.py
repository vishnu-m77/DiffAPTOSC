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
    def __init__(self, data_path, total_image_num, train=True):
        self.trainsize = (224, 224)
        #self.trainsize = (100, 100)
        self.total_image_num = total_image_num
        self.train = train
        self.data_path = data_path

        with open(data_path, "rb") as f:
            tr_dl = json.load(f)
        self.dataset = tr_dl

        # print(self.size)
        if train:
            logging.info("Initialize DataLoader for train dataset")
            '''
            Lakshay - quick update:
            now selecting randomly generated n number of images. n is mentioned in total_images in params.json
            mulitplier 0.7 mentioned to make sure we select 0.7 * n images from all training images
            '''
            self.dataset = random_image_selection(
                self.dataset, 0.7, total_image_num)
            self.size = len(self.dataset)
            self.transform_center = transforms.Compose([
                tr.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                # tr.CenterCrop(self.trainsize),
                tr.RandomHorizontalFlip(),
                tr.RandomVerticalFlip(),
                tr.RandomRotation(30),
                # tr.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])
        else:
            logging.info("Initialize DataLoader for test dataset")
            '''
            Lakshay - quick update:
            now selecting randomly generated n number of images. n is mentioned in total_images in params.json
            mulitplier 0.3 mentioned to make sure we select 0.3 * n images from all testing images
            '''
            self.dataset = random_image_selection(
                self.dataset, 0.3, total_image_num)
            self.size = len(self.dataset)
            self.transform_center = transforms.Compose([
                tr.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                # trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

        # self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        '''# no idea what the index does ...

        * edit (Lakshay) - What this function helps us to is get a single element of the object
        This function is invoked when someone does something like this:
        demo = APTOSDataset(data_path, True)
        print(demo[1]) <- this is when this function is invoked automatically
        '''
        data_pac = self.dataset[index]
        img_path = data_pac['img_root']
        # cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')
        """
        Sehmi - tried experimenting with the transformations below.
        Please do not remove commented section below until I do.

        * Important Note:
        The authors use transforms.ToTensor() insead of transforms.PILToTensor().

        Now if I do transforms.PILToTensor() on an image object to get image_torch, I'll obtain a tensor with dtype uint8
        and the numbers will be 0-255. Then Image.fromarray(np.asarray(image_torch) gives me the original image

        If I however do transforms.ToTensor() on an image object, I'll obtain a tensor with dtype float32
        and numbers will be 0-1. Then Image.fromarray(np.asarray(image_torch) does not give me the original image

        This note might come in handy in future.
        More information: https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/ 
        """
        # ts = transforms.Compose([transforms.PILToTensor()]) # test transfor
        # image_torch = ts(img)

        # #print(image_torch.size())
        # sve = Image.fromarray(np.asarray(image_torch).T, 'RGB')
        # print("imag is {}".format(img))
        # print("image_torch {}".format(image_torch))
        # sve.save('img_dataset.png')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        return img_torch, label

    def __len__(self):
        '''
        Lakshay - This works when len(object_name) is invoked, basically gives us length of the object
        '''
        return self.size


class DataProcessor():
    def __init__(self, config, database_name="APTOS"):
        super(DataProcessor, self).__init__()
        self.database_name = database_name
        self.train_path = config["train_path"]
        self.test_path = config["test_path"]
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.total_image_num = config["num_images"]

    def get_dataloaders(self):
        train_data = APTOSDataset(
            data_path=self.train_path, total_image_num=self.total_image_num, train=True)
        test_data = APTOSDataset(
            data_path=self.test_path, total_image_num=self.total_image_num, train=False)

        # print to see if the number of dataset selection works
        # print(len(train_data), len(test_data))

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

        return train_loader, test_loader


def random_image_selection(original, total_image_num, multiplier):
    '''
    Lakshay - 
    original -> contains the list of all the image locations and the labels from either test or train json
    multiplier -> 0.7 for training set and 0.3 for test set
    functionality-> it converts the original list to n * multiplier list. basically it randomly chooses
                    n * multiplier number of image locations and corresponding labels from original
                    and then returns the new_random or new randomly generated set. thus training and testing
                    will happend on total n images and not 3663 images. the number of total images n, is
                    mentioned in params.json, so make changes there.
    '''
    # with open("param/params.json") as paramfile:
    #     random_n = json.load(paramfile)["total_images"]["num"]
    new_random = random.sample(original, int(multiplier*total_image_num))
    # print(new_random)
    return new_random


if __name__ == '__main__':
    ''' 
    Temporarily dumping json file here to test dataloader
    The dataloader module should instead be called from a different file
    '''

    with open("param/params.json") as paramfile:
        param = json.load(paramfile)

    data = DataProcessor(param)
    train_loader, test_loader = data.get_dataloaders()

    """
    Sehmi - tried experimenting with the transformations below.
    Please do not remove commented section below until I do.
    * You can uncomment section below to inspect images of the dataset

    * edit (Lakshay) - changed pickle to json format, to view image locations with ease
    """

    # this image variable conatins all the image locations and the labels associated with them
    # with open("dataset/aptos/aptos_train.json") as paramfile:
    #     image = json.load(paramfile)

    # with open('param/params.json') as paramfile:
    #     num = json.load(paramfile)["total_images"]["num"]

    # image_random_n = random_image_selection(image, num, multiplier=0.4)

    # print(len(image_random_n))
    # img_path = image_random_n[1]['img_root']
    # img = Image.open(img_path).convert('RGB')
    # print(img_path)
    # img.save("img1.png")

    # # train loader and train_labels produced by the dataset object can be used to train the dcg
    train_features, train_labels = next(iter(train_loader))
    # test_features, test_labels = next(iter(test_loader))

    '''
    Lakshay - tried experimenting with this file below.
    Please do not remove commented section below until I do.

    train_features contains the values of 224*224 images in RGB for 32 images, since the batch_size in 
    params.json is 32 and DataLoader uses it above.
    train_features contains the values of true targets for the 32 images 

    Uncomment below to see the size of the train_features and train_labels
    '''
    # print(train_features[1], train_labels[1])
    # print(train_labels.size(), train_features.size(), train_loader)
    # print(test_features[1], test_labels[1])
    # print(test_labels.size(), test_features.size(), test_loader)

    """
    Code below does not produce correct image. More info in __get_item__ method
    
    * edit (Lakshay) - The reason for this is because this array contains the image after the 
    transforms are made to it, that is rotating, flipping, normalizing etc."""
    # img = Image.fromarray(np.asarray(
    #     train_features[1]).T, 'RGB')  # The transpose is probably needed due to RGB and Tensor conversions
    # img.save('out.png')
