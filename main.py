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

if __name__ == '__main__':
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Iterative Retraining')
    
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
    if verbose:
        print(params)
    
    # Creates a report file
    report_file = 'report.txt'
    
    if os.path.exists(report_file):
        os.remove(report_file)
    f = open(report_file, 'w')
    f.close()
    