import torch
import logging
import json
import os
import sys

import torch.nn as nn
import numpy as np
import DCG.dcg_utils as utils
import DCG.dcg_networks as net

class DCG(nn.Module):
    def __init__(self, parameters):
        super(DCG, self).__init__()

if os.path.exists('dcg.log'):
    os.remove('dcg.log')

logging.basicConfig(filename='dcg.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.warning('This will get logged to a file')

if __name__ == '__main__':
    
    # Hyperparameters from json file
    with open("param/params.json") as paramfile:
        param = json.load(paramfile)
    params = param["dcg"]
    