import torch
import numpy as np
import time
import os
import src.DCG.main as dcg_module
import src.unet_model as unet_model
import logging
from src.metrics import *

class DiffusionBaseUtils():
    def __init__(self, params):
        super(DiffusionBaseUtils, self).__init__()
        self.T = params['timesteps']  # Number of diffusion steps
        self.noise_schedule = params["noise_schedule"]  # noise schedule
        self.beta_initial = params["beta_initial"]  # initial level of noise
        self.beta_final = params["beta_final"]  # final level of noise

    @property
    def noise_list(self):
        """
        Returns a list containing noise to be added in the forward diffusion process. The list is an increasing sequence
        because in diffusion we generally start by small perturbations on data and then keep on increasing the perturbations 
        """
        if self.noise_schedule == "Linear":
            betas = torch.linspace(self.beta_initial, self.beta_final, self.T)
        else:
            raise NotImplementedError("{0} noise schedule not implemented".format(self.noise_schedule))
        return betas

    def get_alpha_prod(self, timestep=None):
        """
        Returns alpha_prod which is the product of alpha_t where
        alpha_t = 1 - beta_t for all time until timestep
        """
        if timestep == None:
            timestep = self.T
        alphas = 1-self.noise_list[:timestep]
        alpha_prod = torch.prod(alphas)
        return alpha_prod

    def reverse_diffusion_parameters(self, t):
        """
        Returns parameters which are used in the reverse diffusion process of the 
        CARD paper
        """
        beta_t = self.noise_list[t]  # beta_t at timestep t
        alpha_prod_t = self.get_alpha_prod(timestep=t)

        if t < 1:
            raise ValueError("Invalid timestep. Timestep must be at least 1 to obtain reverse diffusion parameters")
        alpha_prod_t_m1 = self.get_alpha_prod(timestep=t-1)

        gamma_0 = beta_t*(torch.sqrt(alpha_prod_t_m1)/(1 - alpha_prod_t))
        gamma_1 = ((1-alpha_prod_t_m1)*torch.sqrt(1-beta_t))/(1-alpha_prod_t)
        gamma_2 = 1+((torch.sqrt(alpha_prod_t)-1)*(torch.sqrt(1-beta_t)+torch.sqrt(alpha_prod_t_m1)))/(1-alpha_prod_t)
        beta_var = ((1-alpha_prod_t_m1)*beta_t)/(1-alpha_prod_t)

        return gamma_0, gamma_1, gamma_2, beta_var, alpha_prod_t


class ForwardDiffusion(DiffusionBaseUtils):
    def __init__(self, params):
        super(ForwardDiffusion, self).__init__(params=params)

    def forward(self, var, prior, eps=None, t=None):
        """
        This method is used to add noise to y_0, global_prior and local prior and then 
        obtain the respective noisy variables following equation 2 of DiffMIC paper.
        y_0, global and local priors will be obtained from dcg.

        - t is the timestep till which noise has been added in the forward diffusion process

        - var is the variable on which we are adding noise and it needs to be a float tensor

        Note:
        * prior = (global_prior + local_prior)/2 for y_0
        * prior = global_prior for y_global
        * prior = local_prior for y_local
        """
        if t == None:
            t = self.T  # If no t is defined, add noise till timestep given in params file
        if eps == None:
            eps = torch.randn_like(var)  # gaussian noise
        # generate alpha_prod for t time (where t is time for which noise has been added)
        alpha_prod = self.get_alpha_prod(timestep=t)
        noised_var = torch.sqrt(alpha_prod)*var + torch.sqrt(1-alpha_prod)*eps + (
            1-torch.sqrt(alpha_prod))*prior  # add noise till timestep = T

        return noised_var


class ReverseDiffusion(DiffusionBaseUtils):
    def __init__(self, params):
        super(ReverseDiffusion, self).__init__(params=params)

    def reverse_diffusion_step(self, x, y_t, t, cond_prior, score_net):
        # First we calculate the time dependent parameters gamma_0, gamme_1, gamma_2 and beta_var
        # In reverse diffusion, at each timestep t, we are essentially sampling from a Gaussian Distribution
        # whose mean is defined using gamma_0, gamma_1 and gamma_2 and the variance is defined by beta_var
        # Note that gamma_0, gamma_1, gamma_2, beta_var depend on the timestep of reverse diffusion
        gamma_0, gamma_1, gamma_2, beta_var, alpha_prod_t = self.reverse_diffusion_parameters(t)
        eps = torch.randn_like(y_t)
        t = torch.tensor([t])

        # Now we reparameterize y0 to obtain y0_hat
        y0_hat = (1/torch.sqrt(alpha_prod_t))*(y_t - (1-torch.sqrt(alpha_prod_t)) *
                                               cond_prior - torch.sqrt(1-alpha_prod_t)*score_net.forward(x, y_t, t, yhat=cond_prior))

        y_tm1 = gamma_0*y0_hat+gamma_1*y_t+gamma_2 * cond_prior+torch.sqrt(beta_var)*eps

        return y_tm1, y0_hat

    def full_reverse_diffusion(self, x, cond_prior, score_net, t1, t2, t3):
        """
        This does the full reverse diffusion process. We start by initializing a random sample from a Gaussian Distribution
        whose mean is defined by the cond_prior and with variance I. Then reverse_diffusion_step is called self.T - 1 times.
        In the very last step i.e. at time = 1,
        """

        y_t = torch.rand_like(cond_prior)+cond_prior
        for t in range(self.T):
            t = self.T - t - 1
            if t > 0:
                y_tm1, y0_hat = self.reverse_diffusion_step(x, y_t, t, cond_prior, score_net) 
                y_t = y_tm1

                if t == t1:
                    y_t_1 = y_tm1
                elif t == t2:
                    y_t_2 = y_tm1
                elif t == t3:
                    y_t_3 = y_tm1
            else:
                y_tm1 = y0_hat

        y0_synthetic = y_tm1
        return y0_synthetic, y_t_1, y_t_2, y_t_3


def compute_kernel(x, y):
    """
    Helper function to compute MMD loss
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  
    y = y.unsqueeze(0) 
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)


def compute_mmd(x, y):
    """
    Computes MMD Loss
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel + y_kernel - 2*xy_kernel
    return mmd


class weighted_loss():
    """
    Class used to compute weighted loss
    """
    def __init__(self, y, weight = "n"):
        self.y = y
        w = [1/1805, 1/370, 1/999, 1/193, 1/295]
        w = np.asarray(w)
        w = torch.from_numpy(w)
        if weight == "None":
            self.weights = None
        elif weight == "n":
            self.weights = w
        elif weight == "sqrt":
            w1 = torch.sqrt(w)
            self.weights = w1
        else:
            logging.error("Unknown value given to weight parameter in diffusion")
            raise KeyError("Unknown value given to weight parameter in diffusion")

    def loss(self, loss_vector):
        if self.weights != None:
            weight_list = []
            for label in self.y:
                label = self.weights[label]
                weight_list.append(label)
            loss_vector = torch.sum(loss_vector, dim=1)
            loss_vector = loss_vector*torch.tensor(weight_list)
        return loss_vector


def get_loss(x, y, params, dcg, FD, model):
    """
    Helper function to compute loss
    """
    y0, _ = dcg.cast_label_to_one_hot_and_prototype(y)
    n = x.size(0)
    num_timesteps = params["timesteps"]
    t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,))
    t = torch.cat([t, num_timesteps - 1 - t], dim=0)[:n]
    dcg_fusion, dcg_global, dcg_local = dcg.forward(x)
    dcg_fusion = dcg_fusion.softmax(dim=1)
    dcg_global, dcg_local = dcg_global.softmax(dim=1), dcg_local.softmax(dim=1)
    eps = torch.randn_like(y0)
    # Creates noise with the priors
    yt_fusion = FD.forward(y0, dcg_fusion, eps=eps)
    yt_global = FD.forward(y0, dcg_global, eps=eps)
    yt_local = FD.forward(y0, dcg_local, eps=eps)
    output = model(x, yt_fusion, t, dcg_fusion)
    output_global = model(x, yt_global, t, dcg_global)
    output_local = model(x, yt_local, t, dcg_local)

    weighted = weighted_loss(y=y, weight=params["weight"])
    loss = weighted.loss((eps - output).square()).mean() + 0.5*(weighted.loss(compute_mmd(eps, output_global)).mean() +
                                                                weighted.loss(compute_mmd(eps, output_local)).mean())

    return loss


def train(dcg, model, params, train_loader, val_loader):
    """
    Main training function which contains training loop
    """
    FD = ForwardDiffusion(params=params)  # initialize class

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0033, betas=(0.9, 0.999), amsgrad=False, weight_decay=0.00, eps=0.00000001)
    data_start = time.time()
    data_time = 0
    train_epoch_num = params["num_epochs"]
    val_iter = iter(val_loader)
    loss_batch = []
    loss_batch_val = []
    for epoch in range(0, train_epoch_num):
        for i, feature_label_set in enumerate(train_loader):
            # load images and labels from train dataset
            x_batch, y_labels_batch = feature_label_set
            # load images and labels from val dataset
            try:
                x_val, y_labels_val = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_val, y_labels_val = next(val_iter)
            model.train()
            loss = get_loss(x_batch, y_labels_batch, params, dcg, FD, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())

            # eval
            model.eval()
            val_loss = get_loss(x_val, y_labels_val, params, dcg, FD, model)
            loss_batch_val.append(val_loss.item())

            logging.info(f"epoch: {epoch+1}, batch {i+1} Diffusion training loss: {loss}\t validation loss: {val_loss}")

    data_time = time.time() - data_start
    logging.info("\nTraining of Diffusion took {:.4f} minutes.\n".format((data_time) / 60))
    # save DCG model after training
    diff_states = [
        model.state_dict(),
        optimizer.state_dict(),
    ]
    torch.save(diff_states, "saved_diff.pth")
    plot_loss(loss_arr=loss_batch, val_loss_array=loss_batch_val, mode="diffusion")

def eval(dcg, model, params, test_loader):
    """
    Main eval function which contains evaluation loop
    """
    t1, t2, t3 = params["t_sne"]["t1"], params["t_sne"]["t2"], params["t_sne"]["t3"]
    reverse_diffusion = ReverseDiffusion(params=params)
    targets = []
    dcg_output = []
    diffusion_output = []
    y_outs = []
    y_outs_1 = []
    y_outs_2 = []
    y_outs_3 = []

    model.eval()
    for i, feature_label_set in enumerate(test_loader):
        x_batch, y_labels_batch = feature_label_set
        dcg_fusion, _, _ = dcg.forward(x_batch)
        dcg_fusion = dcg_fusion.softmax(dim=1)  # the actual label
        y_T_mean = dcg_fusion
        y_out, y_out_1, y_out_2, y_out_3 = reverse_diffusion.full_reverse_diffusion(
            x_batch, cond_prior=y_T_mean, score_net=model, t1=t1, t2=t2, t3=t3)
        logging.info("Actual: {}, DCG_out: {}, Diff_out: {}".format(
            y_labels_batch, torch.argmax(y_T_mean, dim=1), torch.argmax(y_out.softmax(dim=1), dim=1)))
        targets.append(y_labels_batch)
        dcg_output.append(torch.argmax(y_T_mean, dim=1))
        diffusion_output.append(torch.argmax(y_out.softmax(dim=1), dim=1))

        for inner_array in y_out:
            y_outs.append(inner_array.detach().numpy())
        for inner_array in y_out_1:
            y_outs_1.append(inner_array.detach().numpy())
        for inner_array in y_out_2:
            y_outs_2.append(inner_array.detach().numpy())
        for inner_array in y_out_3:
            y_outs_3.append(inner_array.detach().numpy())

    targets = torch.cat(targets)
    dcg_output = torch.cat(dcg_output)
    diffusion_output = torch.cat(diffusion_output)
    y = np.stack((y_outs, y_outs_1, y_outs_2, y_outs_3), axis=0)

    return targets, dcg_output, diffusion_output, y
