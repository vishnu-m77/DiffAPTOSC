import torch
import os
import sys
import argparse
import json
import logging

import src.dataloader.dataloader as dataloader
import src.DCG.main as dcg_module
import src.diffusion as diffusion
import src.unet_model as unet_model
import src.metrics as metrics

logging.getLogger('matplotlib').setLevel(logging.WARNING)

if os.path.exists('project.log'):
    os.remove('project.log')

logging.basicConfig(filename='project.log', filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(device)

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
    unet_params = param["unet"]

    if verbose:
        logging.info('params are {}'.format(param))

    data = dataloader.DataProcessor(data_params)
    train_loader, test_loader, val_loader = data.get_dataloaders()
    
    y_fusions = []
    y_globals = []
    y_locals = []

    dcg_chkpt_path = "saved_dcg.pth"
    # Checks if there is a saved DCG checkpoint. If not, trains the DCG.
    if not os.path.exists(dcg_chkpt_path):
        # Initialize DCG
        dcg = dcg_module.DCG(dcg_params)
        # Trains DCG and saves the model
        dcg_module.train_DCG(dcg, dcg_params, train_loader, val_loader=val_loader)
    # Loads the saved DCG model and sets to eval mode
    logging.info(
        "Loading trained DCG checkpoint from {}".format(dcg_chkpt_path))
    dcg, optimizer = dcg_module.load_DCG(dcg_params)
    logging.info("DCG completed")

    if verbose:
        logging.info("Diffusion model parameters: {}".format(diffusion_params))

    model = unet_model.ConditionalModel(
        config=unet_params, n_steps=diffusion_params["timesteps"], n_classes=data_params["num_classes"], guidance=diffusion_params["include_guidance"]).to(device)
    diff_chkpt_path = 'saved_diff.pth'
    # Checks if a saved diffusion checkpoint exists. If not, trains the diffusion model.
    if not os.path.exists(diff_chkpt_path):
        diffusion.train(dcg, model, diffusion_params, train_loader, val_loader=val_loader)

    logging.info(
        "Loading trained diffusion checkpoint from {}".format(diff_chkpt_path))
    chkpt = torch.load(diff_chkpt_path)
    model.load_state_dict(chkpt[0])
    model.eval()
    logging.info("Diffusion_checkpoint loaded")
    targets, dcg_output, diffusion_output, y = diffusion.eval(dcg, model, diffusion_params, test_loader)
    
    metrics.call_metrics(diffusion_params, targets, dcg_output, diffusion_output, y)
