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

    for epoch in range(0, 5):

        data_start = time.time()
        data_time = 0

        for i, feature_label_set in enumerate(train_loader):

            x_batch, y_labels_batch = feature_label_set
            y_one_hot_batch, y_logits_batch = dcg_module.cast_label_to_one_hot_and_prototype(
                y_labels_batch, param)

            n = x_batch.size(0)

            # record unflattened x as input to guidance aux classifier
            # the below 3 lines should be un-commented if we have model.arch in ["simple", "linear"]
            x_unflat_batch = x_batch
            # x_unflat_batch = x_batch.to(self.device)
            # if param.data.dataset == "toy" or param.model.arch in ["simple", "linear"]:
            #     x_batch = torch.flatten(x_batch, 1)

            data_time += time.time() - data_start

            model.train()

            dcg.eval()

            # step += 1
            # antithetic sampling
            num_timesteps = param["diffusion"]["timesteps"]
            t = torch.randint(low=0, high=num_timesteps,
                              size=(n // 2 + 1,))
            t = torch.cat([t, num_timesteps - 1 - t], dim=0)[:n]

            # noise estimation loss
            # x_batch = x_batch.to(self.device)
            # y_0_batch = y_logits_batch.to(self.device)
            y_0_hat_batch, y_0_global, y_0_local = dcg(x_unflat_batch)
            y_0_hat_batch = y_0_hat_batch.softmax(dim=1)
            y_0_global, y_0_local = y_0_global.softmax(
                dim=1), y_0_local.softmax(dim=1)

            # print(y_0_hat_batch.size(), y_0_global.size(), y_0_local.size())

            y_T_mean = y_0_hat_batch
            # if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
            #     y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)
            y_0_batch = y_one_hot_batch.to(self.device)
            e = torch.randn_like(y_0_batch).to(y_0_batch.device)
            y_t_batch = q_sample(y_0_batch, y_T_mean,
                                 self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
            y_t_batch_global = q_sample(y_0_batch, y_0_global,
                                        self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
            y_t_batch_local = q_sample(y_0_batch, y_0_local,
                                       self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
            # output = model(x_batch, y_t_batch, t, y_T_mean)
            output = model(x_batch, y_t_batch, t, y_0_hat_batch)
            output_global = model(x_batch, y_t_batch_global, t, y_0_global)
            output_local = model(x_batch, y_t_batch_local, t, y_0_local)

    ####################  Reverse diffusion code end   #############################

    if os.path.exists(report_file):
        os.remove(report_file)
    f = open(report_file, 'w')
    f.close()
