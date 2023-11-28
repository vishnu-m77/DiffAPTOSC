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
from src.diffusion import ReverseDiffusion, ForwardDiffusion, compute_mmd

import src.dataloader.dataloader as dataloader
import src.DCG.main as dcg_module
import src.diffusion as diffusion
import src.unet_model as unet_model

import matplotlib.pyplot as plt



logging.getLogger('matplotlib').setLevel(logging.WARNING)

if os.path.exists('project.log'):
    os.remove('project.log')

logging.basicConfig(filename='project.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
# logging.warning('This will get logged to a file')

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
    dataset = args.dataset

    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)

    data_params = param["data"]
    dcg_params = param["dcg"]
    diffusion_params = param['diffusion']

    # The following 4 lines of code use dataset_params. If we don't need these, we can delete it but keep a copy for now.
    # dataset_params = param[dataset]
    # MODEL_VERSION_DIR = "diffmic_conditional_results/" + str(dataset_params['N_STEPS']) + "steps/nn/" + str(
    #     dataset_params["RUN_NAME"]) + "/" + str(dataset_params["PRIOR_TYPE"]) + str(dataset_params["CAT_F_PHI"]) + "/" + str(dataset_params["F_PHI_TYPE"])
    # dataset_params["MODEL_VERSION_DIR"] = MODEL_VERSION_DIR
    
    # logging.debug('verbose is {}'.format(verbose))
    if verbose:
        logging.info('params are {}'.format(param))
        # print(data_params)

    # Creates a report file
    report_file = 'report.txt'

    data = dataloader.DataProcessor(data_params)
    train_loader, test_loader = data.get_dataloaders()

    y_fusions = []
    y_globals = []
    y_locals = []
    
    dcg_chkpt_path = "saved_dcg.pth"
    # Checks if there is a saved DCG checkpoint. If not, trains the DCG.
    if not os.path.exists(dcg_chkpt_path):
        # Initialize DCG
        dcg = dcg_module.DCG(dcg_params)
        # Trains DCG and saves the model
        dcg_module.train_DCG(dcg, dcg_params, train_loader)
    
    # Loads the saved DCG model and sets to eval mode
    logging.info("Loading trained DCG checkpoint from {}".format(dcg_chkpt_path))
    dcg, optimizer = dcg_module.load_DCG(dcg_params)
    logging.info("DCG completed")

    if verbose:
        logging.info("Diffusion model parameters: {}".format(diffusion_params))
    
    # forward diffusion EXAMPLE call below where the parameters are explained in difusion.py script
    # noised_var = FD.forward(var=torch.tensor(0.0), prior=torch.tensor(0))
    # logging.info("Noised Variable is {}".format(noised_var))

    #################### Reverse diffusion code begins #############################
    model = unet_model.ConditionalModel(config=param, guidance=False)
    diff_chkpt_path = 'saved_diff.pth'
    # Checks if a saved diffusion checkpoint exists. If not, trains the diffusion model.
    if not os.path.exists(diff_chkpt_path):
        diffusion.train(dcg, model, diffusion_params, train_loader)
    
    logging.info("Loading trained diffusion checkpoint from {}".format(diff_chkpt_path))
    chkpt = torch.load(diff_chkpt_path)
    model.load_state_dict(chkpt[0])
    model.eval()
    logging.info("Diffusion_checkpoint loaded")
    diffusion.eval(dcg, model, diffusion_params, test_loader)

    #################### Reverse diffusion code ends #############################

    if os.path.exists(report_file):
        os.remove(report_file)
    f = open(report_file, 'w')
    f.close()
