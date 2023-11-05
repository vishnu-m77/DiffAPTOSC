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

import src.dataloader.dataloader as dataloader
import src.DCG.main as dcg_module
import src.diffusion as diffusion

if os.path.exists('project.log'):
    os.remove('project.log')

logging.basicConfig(filename='project.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.warning('This will get logged to a file')

if __name__ == '__main__':
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='DiffMIC')
    
    # Default values of parameters are defined
    parser.add_argument('--param', default = 'param/params.json', help='file containing hyperparameters')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    parser.add_argument('-d', '--dataset', default = 'aptos', type = str, help='dataset')
    
    args = parser.parse_args()
    verbose = args.verbose
    data = args.dataset
    
    # Hyperparameters from json file
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    
    params = param[data]
    MODEL_VERSION_DIR = "diffmic_conditional_results/" + str(params['N_STEPS']) + "steps/nn/" + str(params["RUN_NAME"]) + "/" + str(params["PRIOR_TYPE"]) + str(params["CAT_F_PHI"]) + "/" + str(params["F_PHI_TYPE"])
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
    y_fusions = []
    y_globals = []
    y_locals = []
    for ind, (image, target) in enumerate(train_loader):
        # x = torch.flatten(x, 1)
        # print(image)
        y_fusion, y_global, y_local = dcg.forward(image)
        y_fusions.append(y_fusion)
        y_locals.append(y_local)
        y_globals.append(y_global)
        # logging.info(y_global)
    
    logging.info("DCG completed")
    
    # diff = diffusion.ForwardDiffusionUtils()
    # for i in y_fusions:
    #     noised_var = diff.forward(var, y_fusion, t)
        
    
    
    
    if os.path.exists(report_file):
        os.remove(report_file)
    f = open(report_file, 'w')
    f.close()
    