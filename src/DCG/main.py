import torch
import logging
import json
import os
import sys

import torch.nn as nn
import numpy as np
import src.DCG.utils as utils
import src.DCG.networks as net

# This implementation uses helper functions to define the intermediate 'layers' of DCG instead of classes.

class DCG(nn.Module):
    def __init__(self, parameters):
        super(DCG, self).__init__()

        logging.info("Initialize DCG")
        # save parameters
        self.parameters = parameters

    def forward(self, x_original):
        """
        :param x_original: N,H,W,C numpy matrix
        """
        # global network: x_small -> class activation map
        # h_g, self.saliency_map = self.global_network.forward(x_original)
        h_g, self.saliency_map = net.global_res.forward(x_original, self.parameters)

        # calculate y_global
        # note that y_global is not directly used in inference
        # self.y_global = self.aggregation_function.forward(self.saliency_map)
        self.y_global = net.aggregator(self.parameters["percent_t"], self.saliency_map)

        # a region proposal network
        # small_x_locations = utils.retrieve_roi_crops.forward(x_original, self.cam_size, self.saliency_map)
        small_x_locations = net.retrieve_roi(x_original, self.parameters["cam_size"], self.saliency_map, self.parameters)

        # convert crop locations that is on self.parameters["cam_size"] to x_original
        self.patch_locations = utils._convert_crop_position(small_x_locations, self.parameters["cam_size"], x_original)

        # patch retriever
        crops_variable = utils._retrieve_crop(x_original, self.patch_locations, self.parameters["crop_method"], self.parameters)
        self.patches = crops_variable.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = net.local_res(crops_variable).view(batch_size, num_crops, -1)

        # MIL module
        # y_local is not directly used during inference
        # z, self.patch_attns, self.y_local = net.attention(h_crops, self.parameters)
        self.y_local = net.attention(h_crops, self.parameters)

        self.y_fusion = 0.5* (self.y_global+self.y_local)
        return self.y_fusion, self.y_global, self.y_local


if __name__ == '__main__':
    if os.path.exists('dcg.log'):
        os.remove('dcg.log')

    logging.basicConfig(filename='dcg.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.warning('This will get logged to a file')
    
    # Hyperparameters from json file
    with open("../../param/params.json") as paramfile:
        param = json.load(paramfile)
    params = param["dcg"]
    dcg = DCG(params)
    