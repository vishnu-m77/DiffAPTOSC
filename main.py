import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.utils as utils
import torch.multiprocessing as mp

import math as m
import random
import os
import sys
import time

# import psutil
import argparse
import json
# from itertools import product, permutations, combinations
import tqdm
import argparse
import traceback
import shutil
import logging
from src.diffusion import ReverseDiffusion, ForwardDiffusion

import src.dataloader.dataloader as dataloader
import src.DCG.main as dcg_module
import src.diffusion as diffusion

from src.network.unet_model import *

if os.path.exists('project.log'):
    os.remove('project.log')

logging.basicConfig(filename='project.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.warning('This will get logged to a file')

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='DiffMIC')

    # Default values of parameters are defined
    parser.add_argument('--param', default='param/params.json',
                        help='file containing hyperparameters')
    parser.add_argument('-v', '--verbose',
                        help='increase output verbosity', action='store_true')
    parser.add_argument('-d', '--dataset', default='aptos',
                        type=str, help='dataset')

    args = parser.parse_args()
    verbose = args.verbose
    data = args.dataset

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    params = param[data]
    MODEL_VERSION_DIR = "diffmic_conditional_results/" + str(params['N_STEPS']) + "steps/nn/" + str(
        params["RUN_NAME"]) + "/" + str(params["PRIOR_TYPE"]) + str(params["CAT_F_PHI"]) + "/" + str(params["F_PHI_TYPE"])
    params["MODEL_VERSION_DIR"] = MODEL_VERSION_DIR
    # logging.debug('verbose is {}'.format(verbose))
    if verbose:
        logging.info('params are {}'.format(params))
        # print(params)

    # Creates a report file
    report_file = 'report.txt'

    data = dataloader.DataProcessor(param)
    train_loader, test_loader = data.get_dataloaders()

    dcg_params = param["dcg"]
    dcg = dcg_module.DCG(dcg_params)
    dcg_train = False
    if dcg_train == True:
        dcg_module.train_DCG(dcg, param, train_loader, test_loader)
    y_fusions = []
    y_globals = []
    y_locals = []
    # for ind, (image, target) in enumerate(train_loader):
    #     # x = torch.flatten(x, 1)
    #     # print(image)
    #     y_fusion, y_global, y_local = dcg.forward(image)
    #     y_fusions.append(y_fusion)
    #     y_locals.append(y_local)
    #     y_globals.append(y_global)
    # logging.info(y_global)

    logging.info("DCG completed")

    diffusion_config = param['diffusion']
    if verbose:
        logging.info("Diffusion model parameters: {}".format(diffusion_config))
    FD = ForwardDiffusion(config=diffusion_config)  # initialize class
    # forward diffusion EXAMPLE call below where the parameters are explained in difusion.py script
    noised_var = FD.forward(var=torch.tensor(0.0), prior=torch.tensor(0))
    logging.info("Noised Variable is {}".format(noised_var))

    #################### Reverse diffusion code begins #############################
    model = ConditionalModel(config=param, guidance=False)

    dcg.load_state_dict(torch.load('aux_ckpt.pth')[0])
    dcg.eval()
    reverse_diffusion = ReverseDiffusion(config=diffusion_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for epoch in range(0, 50):

        data_start = time.time()
        data_time = 0

        for i, feature_label_set in enumerate(train_loader):

            x_batch, y_labels_batch = feature_label_set
            y_one_hot_batch, y_logits_batch = dcg_module.cast_label_to_one_hot_and_prototype(
                y_labels_batch, param)

            n = x_batch.size(0)

            num_timesteps = param["diffusion"]["timesteps"]
            t = torch.randint(low=0, high=num_timesteps,
                              size=(n // 2 + 1,))
            t = torch.cat([t, num_timesteps - 1 - t], dim=0)[:n]
            #print(t)

            dcg_fusion, dcg_global, dcg_local = dcg(x_batch)[0], dcg(x_batch)[1], dcg(x_batch)[2]
            """
            sehmi -  why softmax in the 2 line below?
            Note sure, but it seems to work...
            """
            dcg_fusion = dcg_fusion.softmax(dim=1)
            dcg_global,dcg_local = dcg_global.softmax(dim=1),dcg_local.softmax(dim=1)
            #print(dcg_global)
            y0 = y_one_hot_batch
            eps = torch.randn_like(y0)
            yt_fusion = FD.forward(y0, dcg_fusion, eps=eps)
            yt_global = FD.forward(y0, dcg_global, eps=eps)
            yt_local = FD.forward(y0, dcg_local, eps=eps)
            output = model(x_batch, yt_fusion, t, dcg_fusion)
            output_global = model(x_batch, yt_global, t, dcg_global)
            output_local = model(x_batch, yt_local, t, dcg_local)
            #print(output_global)
            loss = (eps - output).square().mean()# + 0.5*(compute_mmd(eps,output_global) + compute_mmd(eps,output_local))
            optimizer.zero_grad()
            loss.backward()
            print(loss.item())

    if os.path.exists(report_file):
        os.remove(report_file)
    f = open(report_file, 'w')
    f.close()
