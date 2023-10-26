import torch, cv2
import numpy as np
import pickle, json
import matplotlib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import transforms as trans

"""
Cirrently using transforms defined by authors, but we can later on
replace functions by using torchvision transforms instead
"""


class APTOSDataset(Dataset):
    def __init__(self, data_path, train=True):
        self.trainsize = (224,224)
        self.train = train
        self.data_path = data_path
        with open(data_path, "rb") as f:
            tr_dl = pickle.load(f)
        self.dataset = tr_dl

        self.size = len(self.dataset)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        ## no idea what the index does ...
        data_pac = self.dataset[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
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
        ts = transforms.Compose([transforms.PILToTensor()]) # test transfor
        image_torch = ts(img)

        #print(image_torch.size())
        sve = Image.fromarray(np.asarray(image_torch).T, 'RGB')
        print("imag is {}".format(img))
        print("image_torch {}".format(image_torch))
        sve.save('img_dataset.png')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label

    def __len__(self):
        return self.size


class DataProcessor():
    def __init__(self, config, database_name = "APTOS"):
        super(DataProcessor,self).__init__()
        self.database_name = database_name
        self.train_path = config["data"]["train_path"]
        self.test_path = config["data"]["test_path"]
        self.train_batch_size =  config["train"]["batch_size"]
        self.test_batch_size =  config["test"]["batch_size"]

    def get_dataloaders(self):
        train_data = APTOSDataset(data_path=self.train_path, train = True)
        test_data = APTOSDataset(data_path=self.test_path, train = False)

        train_loader = DataLoader(
            train_data,
            batch_size=self.train_batch_size,
            shuffle=True            
            #sampler=sampler
        )
        test_loader = DataLoader(
            test_data,
            batch_size=self.test_batch_size,
            shuffle=False
        )

        return train_loader, test_loader


if __name__ == '__main__':
    # temporarily dumping json file here to test dataloader
    # the dataloader module should instead be called fromma different file
    with open("param/params.json") as paramfile:
        param = json.load(paramfile)

    data = DataProcessor(param)
    train_loader, test_loader = data.get_dataloaders()
    """
    Sehmi - tried experimenting with the transformations below.
    Please do not remove commented section below until I do.
    * You can uncomment section below to inspect images of the dataset
    """
    # print(train_data.size())
    # with open("dataset/aptos/aptos_train.pkl", "rb") as f:
    #     tr_dl = pickle.load(f)
    # img_path = tr_dl[0]['img_root']
    # image = Image.open(img_path).convert('RGB')
    # image.save("img.png")
    train_features, train_labels = next(iter(train_loader)) # train loader and train_labels produced by the dataset object can be used to train the dcg
    #print(train_features.size())
    """Code below does not produce correct image. More info in __get_item__ method"""
    # img = Image.fromarray(np.asarray(train_features[1]).T, 'RGB') # The transpose is probably needed due to RGB and Tensor conversions
    # img.save('out.png')

    print(train_labels.size())


    

