import torch
import logging
import json
import os
import sys
import time

import torch.nn as nn
import numpy as np
import src.DCG.utils as utils
import src.DCG.networks as net
from src.metrics import plot_loss

# if os.path.exists('src/DCG/dcg.log'):
#      os.remove('src/DCG/dcg.log')

# logging.basicparams(filename='src/DCG/dcg.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class DCG(nn.Module):
    def __init__(self, params):
        super(DCG, self).__init__()

        logging.info("Initialize DCG")
        # save params
        self.params = params
        # construct networks
        # global network
        self.global_network = net.GlobalNetwork(self.params, self)
        self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = net.TopTPercentAggregationFunction(
            self.params, self)

        # detection module
        self.retrieve_roi_crops = net.RetrieveROIModule(self.params, self)

        # detection network
        self.local_network = net.LocalNetwork(self.params, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = net.AttentionModule(self.params, self)
        self.attention_module.add_layers()
        # fusion branch

    def forward(self, x_original):
        """
        :param x_original: N,H,W,C numpy matrix
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x_original)

        # calculate y_global
        # note that y_global is not directly used in inference
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # a region proposal network

        small_x_locations = self.retrieve_roi_crops.forward(x_original, self.params["cam_size"], self.saliency_map)

        # convert crop locations that is on self.params["cam_size"] to x_original
        self.patch_locations = utils._convert_crop_position(small_x_locations, self.params["cam_size"], x_original)

        # patch retriever
        crops_variable = utils._retrieve_crop(x_original, self.patch_locations, self.params["crop_method"], self.params)
        self.patches = crops_variable.data.cpu().numpy()

        # detection network
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = self.local_network.forward(crops_variable).view(batch_size, num_crops, -1)

        # MIL module
        # y_local is not directly used during inference
        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        self.y_fusion = 0.5 * (self.y_global + self.y_local)
        return self.y_fusion, self.y_global, self.y_local

    def nonlinear_guidance_model_train_step(self, criterion, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred, y_global, y_local = self(x_batch)
        # y_batch_pred = y_batch_pred.softmax(dim=1)
        # aux_cost = self.aux_cost_function(y_batch_pred, y_batch)+self.aux_cost_function(y_global, y_batch)+self.aux_cost_function(y_local, y_batch)
        aux_cost = criterion(y_batch_pred, y_batch)
        # update non-linear guidance model
        if aux_optimizer != None:
            aux_optimizer.zero_grad()
            aux_cost.backward()
            aux_optimizer.step()
        return aux_cost.item()

    def cast_label_to_one_hot_and_prototype(self, y_labels_batch, return_prototype=True):
        """
        y_labels_batch: a vector of length batch_size.
        """
        y_one_hot_batch = nn.functional.one_hot(
            y_labels_batch, num_classes=self.params["num_classes"]).float()
        if return_prototype:
            label_min, label_max = self.params["label_min_max"]
            y_logits_batch = torch.logit(nn.functional.normalize(
                torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
            return y_one_hot_batch, y_logits_batch
        else:
            return y_one_hot_batch


def train_DCG(dcg, params, train_loader, test_loader):

    criterion = nn.CrossEntropyLoss()
    brier_score = nn.MSELoss()
    optimizer = torch.optim.SGD(dcg.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    dcg.train()
    pretrain_start_time = time.time()
    test_iter = iter(test_loader)
    loss_batch, loss_test_array = [], []
    for epoch in range(params["num_epochs"]):
        # loss_batch = []
        for feature_label_set in train_loader:
            # train loss
            dcg.train()
            x_batch, y_labels_batch = feature_label_set
            y_one_hot_batch, y_logits_batch = dcg.cast_label_to_one_hot_and_prototype(y_labels_batch)
            aux_loss = dcg.nonlinear_guidance_model_train_step(criterion, x_batch, y_one_hot_batch, optimizer)
            loss_batch.append(aux_loss)
            # eval loss
            dcg.eval()
            # load images and labels from test dataset
            try:
                x_test, y_labels_test = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                x_test, y_labels_test = next(test_iter)
            y_labels_test_one_hot, _ = dcg.cast_label_to_one_hot_and_prototype(y_labels_test)
            aux_loss_test = dcg.nonlinear_guidance_model_train_step(criterion, x_test, y_labels_test_one_hot, aux_optimizer=None)
            loss_test_array.append(aux_loss_test)

        logging.info(
            f"epoch: {epoch+1}, DCG pre-training loss: {aux_loss} \t test loss: {aux_loss_test}"
        )
    plot_loss(loss_arr=loss_batch, test_loss_array=loss_test_array, title="Loss function for DCG Train",
              savedir='plots/dcg_loss')
    pretrain_end_time = time.time()
    logging.info("\nPre-training of DCG took {:.4f} minutes.\n".format(
        (pretrain_end_time - pretrain_start_time) / 60))
    # save auxiliary model after pre-training
    aux_states = [
        dcg.state_dict(),
        optimizer.state_dict(),
    ]
    torch.save(aux_states, "saved_dcg.pth")

def load_DCG(params):
    model = DCG(params)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    folder = os.getcwd()
    print(folder)
    # Load the saved model state dictionary from the .pth file
    # checkpoint_path = os.path.join(folder, "saved_dcg.pth")
    checkpoint_path = "saved_dcg.pth"
    checkpoint = torch.load(checkpoint_path)
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint[0])
    optimizer.load_state_dict(checkpoint[1])

    # Ensure the model is in evaluation mode
    model.eval()
    return model, optimizer


if __name__ == '__main__':
    if os.path.exists('dcg.log'):
        os.remove('dcg.log')

    logging.basicparams(filename='dcg.log', filemode='a',
                        format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.warning('This will get logged to a file')

    # Hyperparameters from json file
    with open("../../param/params.json") as paramfile:
        param = json.load(paramfile)
    # params = param["dcg"]

    # dcg = DCG(params)
