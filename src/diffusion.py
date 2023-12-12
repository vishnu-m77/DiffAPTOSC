"""
Build basic structure of diffusion pipeline
This script assumes a fully function DCG module and Dataloader module
"""
import torch
import numpy as np
import time
import os
import src.DCG.main as dcg_module
import src.unet_model as unet_model
import logging
# from joblib import Parallel, delayed
from src.metrics import *


class DiffusionBaseUtils():
    def __init__(self, config):
        super(DiffusionBaseUtils, self).__init__()
        self.T = config['timesteps']  # Number of diffusion steps
        self.noise_schedule = config["noise_schedule"]  # noise schedule
        self.beta_initial = config["beta_initial"]  # initial level of noise
        self.beta_final = config["beta_final"]  # final level of noise

    @property
    def noise_list(self):
        """
        Returns a list containing noise to be added in the forward diffusion process. The list is an increasing sequence
        because in diffusion we generally start by small perturbations on data and then keep on increasing the perturbations 
        """
        if self.noise_schedule == "Linear":
            betas = torch.linspace(self.beta_initial, self.beta_final, self.T)
        else:
            raise NotImplementedError(
                "{0} noise schedule not implemented".format(self.noise_schedule))
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
            raise ValueError(
                "Invalid timestep. Timestep must be at least 1 to obtain reverse diffusion parameters")
        # will throw error at time step t = 0 MAKE SURE TO DEAL WITH IT
        alpha_prod_t_m1 = self.get_alpha_prod(timestep=t-1)

        gamma_0 = beta_t*(torch.sqrt(alpha_prod_t_m1)/(1 - alpha_prod_t))
        gamma_1 = ((1-alpha_prod_t_m1)*torch.sqrt(1-beta_t))/(1-alpha_prod_t)
        gamma_2 = 1 + ((torch.sqrt(alpha_prod_t)-1)*(torch.sqrt(1 -
                       beta_t)+torch.sqrt(alpha_prod_t_m1)))/(1-alpha_prod_t)
        beta_var = ((1-alpha_prod_t_m1)*beta_t)/(1-alpha_prod_t)

        return gamma_0, gamma_1, gamma_2, beta_var, alpha_prod_t


class ForwardDiffusion(DiffusionBaseUtils):
    def __init__(self, config):
        super(ForwardDiffusion, self).__init__(
            config=config
        )

    def forward(self, var, prior, eps=None, t=None):
        """
        This method is used to add noise to y_0, global_prior and local prior and then 
        obtain the respective noisy variables following equation 2 of DiffMIC paper.
        y_0, global and local priors will be obtained from dcg.

        t is the timestep till which noise has been added in the forward diffusion process

        var is the variable on which we are adding noise and it needs to be a float tensor

        Note:
        * prior = (global_prior + local_prior)/2 for y_0
        * prior = global_prior for y_global
        * prior = local_prior for y_local
        """
        if t == None:
            t = self.T  # If no t is defined, add noise till timestep given in config file
        if eps == None:
            eps = torch.randn_like(var)  # gaussian noise
        # generate alpha_prod for t time (where t is time for which noise has been added)
        alpha_prod = self.get_alpha_prod(timestep=t)
        noised_var = torch.sqrt(alpha_prod)*var + torch.sqrt(1-alpha_prod)*eps + (
            1-torch.sqrt(alpha_prod))*prior  # add noise till timestep = T

        return noised_var


class ReverseDiffusion(DiffusionBaseUtils):
    def __init__(self, config):
        super(ReverseDiffusion, self).__init__(
            config=config
        )
        """
        NOTES: - Sehmimul - Notes not removed as reverse diffusion notes may be useful in debugging reverse diffusion

        1. The diffusion pipeline in DiffMIC is based off https://arxiv.org/abs/2206.07275 
        Github: https://github.com/XzwHan/CARD/blob/ebe2f2ae95dc7a7a95e6a71c0c8e1cabf8451087/classification/diffusion_utils.py#L106
        Especially the CARD paper's Algorithm 2 line 4 is very useful as thta talks about inference and
        the same inference scheme has been applied in the DiffMIC paper
        Thus CARD has to be an intergal part of our work. Output of p_sample is literally the denoising
        process of the CARD paper.

        2. I understand p_sample, but I don't understand the goal of p_sample_t_1to0. The goal of y_0_reparam is outlined in point 1

        3.  p_sample is just reverse process of CARD paper and it is just equation 9 and Algorithm 2 and it

        4. extract function is not necessary and the x or y input is only required to get the 'shape' and it is not really useful otherwise

        5. In p_sample_loop curl_y is just a sample drawn from the N(prior, I) distribution.
        """

    # cannot test without cond_prior and score_net
    def reverse_diffusion_step(self, x, y_t, t, cond_prior, score_net):
        """
        This is similar to p_sample of code
        x: The input image which will be used in the Score Network
        y_t: noisy variable at a specific timestep t
        t: timestep at which we are doing doing reverse diffusion and t will go from T to 1
        cond_prior: prior for local or global prior or local+global prior and according to CARD it should depend on x
        score_net: Neural Network which is used to approximate the gradient of the log likelihood of the probability distribution
        """
        # First calculate the time dependent parameters gamma_0, gamme_1, gamma_2 and beta_var
        # In reverse diffusion, at each timestep t, we are essentially sampling from a Gaussian Distribution
        # whose mean is defined using gamma_0, gamma_1 and gamma_2 and the variance is defined by beta_var
        # Note that gamma_0, gamma_1, gamma_2, beta_var depend on the timestep of reverse diffusion
        gamma_0, gamma_1, gamma_2, beta_var, alpha_prod_t = self.reverse_diffusion_parameters(
            t=t)
        eps = torch.randn_like(y_t)
        t = torch.tensor([t])

        # first we reparameterize y0 to obtain y0_hat
        y0_hat = (1/torch.sqrt(alpha_prod_t))*(y_t - (1-torch.sqrt(alpha_prod_t)) *
                                               cond_prior - torch.sqrt(1-alpha_prod_t)*score_net.forward(x, y_t, t, yhat=cond_prior))

        y_tm1 = gamma_0*y0_hat+gamma_1*y_t+gamma_2 * \
            cond_prior+torch.sqrt(beta_var)*eps

        return y_tm1, y0_hat

    # cannot test without cond_prior and score_net
    def full_reverse_diffusion(self, x, cond_prior, score_net, t1, t2, t3):
        """
        This does the full reverse diffusion process. We start by initializing a random sample from a Gaussian Distribution
        whose mean is defined by the cond_prior and with variance I. Then reverse_diffusion_step is called self.T - 1 times. In the very last step i.e. at time = 1,
        """

        y_t = torch.rand_like(cond_prior)+cond_prior
        for t in range(self.T):
            # when t = 999, self.T - t - 1 = 0 for self.T = 1000
            # then, from the else condition, we see that we just use the previous y0_hat quantity
            # instead of doing one more reverse diffusion step.
            # The reason I have the - 1 is to follow the convention in the script that we count form 0
            # In card code, they counted from 1 and so here the - 1 ensures that we start counting from 0 and not 1.
            # *** POTENTIAL_BUG ****
            t = self.T - t - 1
            if t > 0:
                y_tm1, y0_hat = self.reverse_diffusion_step(
                    x, y_t, t, cond_prior, score_net)  # method will crash if t = 0
                y_t = y_tm1

                if t == t1:
                    y_t_1 = y_tm1
                elif t == t2:
                    y_t_2 = y_tm1
                elif t == t3:
                    y_t_3 = y_tm1
            else:
                # so t = 1
                y_tm1 = y0_hat

        y0_synthetic = y_tm1
        return y0_synthetic, y_t_1, y_t_2, y_t_3
        # return y0_synthetic


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def get_loss(x,y, params, dcg, FD, model):
    y0, _ = dcg.cast_label_to_one_hot_and_prototype(y)
    n = x.size(0)
    num_timesteps = params["timesteps"]
    t = torch.randint(low=0, high=num_timesteps,
                    size=(n // 2 + 1,))
    t = torch.cat([t, num_timesteps - 1 - t], dim=0)[:n]
    dcg_fusion, dcg_global, dcg_local = dcg.forward(x)
    dcg_fusion = dcg_fusion.softmax(dim=1)
    dcg_global, dcg_local = dcg_global.softmax(
        dim=1), dcg_local.softmax(dim=1)
    eps = torch.randn_like(y0)
    # Creates noise with the priors
    yt_fusion = FD.forward(y0, dcg_fusion, eps=eps)
    yt_global = FD.forward(y0, dcg_global, eps=eps)
    yt_local = FD.forward(y0, dcg_local, eps=eps)
    output = model(x, yt_fusion, t, dcg_fusion)
    output_global = model(x, yt_global, t, dcg_global)
    output_local = model(x, yt_local, t, dcg_local)
    loss = (eps - output).square().mean() + 0.5*(compute_mmd(eps, output_global) + compute_mmd(eps, output_local))
    return loss

def train(dcg, model, params, train_loader, val_loader):
    # model = unet_model.ConditionalModel(config=param, guidance=False)
    FD = ForwardDiffusion(config=params)  # initialize class

    reverse_diffusion = ReverseDiffusion(config=params)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0033, betas=(0.9, 0.999), amsgrad=False, weight_decay=0.00, eps=0.00000001)
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
            loss = get_loss(x_batch,y_labels_batch, params, dcg, FD, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())

            # eval
            model.eval()
            val_loss = get_loss(x_val,y_labels_val, params, dcg, FD, model)
            loss_batch_val.append(val_loss.item())

            logging.info(
                f"epoch: {epoch+1}, batch {i+1} Diffusion training loss: {loss}\t validation loss: {val_loss}")

    data_time = time.time() - data_start
    logging.info("\nTraining of Diffusion took {:.4f} minutes.\n".format(
        (data_time) / 60))
    # save DCG model after training
    diff_states = [
        model.state_dict(),
        optimizer.state_dict(),
    ]
    torch.save(diff_states, "saved_diff.pth")
    plot_loss(loss_arr=loss_batch, val_loss_array=loss_batch_val, mode = False)


def get_out(dcg, model, feature_label_set, reverse_diffusion):
    x_batch, y_labels_batch = feature_label_set
    dcg_fusion, dcg_global, dcg_local = dcg.forward(x_batch)
    dcg_fusion = dcg_fusion.softmax(dim=1)  # the actual label
    y_T_mean = dcg_fusion
    y_out = reverse_diffusion.full_reverse_diffusion(
        x_batch, cond_prior=y_T_mean, score_net=model)
    logging.info("Actual: {}, DCG_out: {}, Diff_out: {}".format(
        y_labels_batch, torch.argmax(y_T_mean, dim=1), torch.argmax(y_out.softmax(dim=1), dim=1)))
    return y_out.softmax(dim=1)


def eval(dcg, model, params, test_loader):
    # dcg.load_state_dict(torch.load('saved_dcg.pth')[0])
    # dcg.eval()
    t1, t2, t3 = params["t_sne"]["t1"], params["t_sne"]["t2"], params["t_sne"]["t3"]
    reverse_diffusion = ReverseDiffusion(config=params)
    # outputs = Parallel(n_jobs=-1)(delayed(self.one_object_pred)(df.loc[df['object_id'] == object], object, report_file, verbose) for object in objects)

    # Parallel/ delayed code calls get_out which does the job of the for loop following it. Only one of the two should be active at any given time
    # outputs = Parallel(n_jobs=-1)(delayed(get_out)(dcg, model, feature_label_set, reverse_diffusion) for i, feature_label_set in enumerate(test_loader))
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
        dcg_fusion, dcg_global, dcg_local = dcg.forward(x_batch)
        dcg_fusion = dcg_fusion.softmax(dim=1)  # the actual label
        y_T_mean = dcg_fusion
        # y_out = reverse_diffusion.full_reverse_diffusion(
        #     x_batch, cond_prior=y_T_mean, score_net=model)
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

        if i+1 >= params['num_test_batches']:
            break
    targets = torch.cat(targets)
    dcg_output = torch.cat(dcg_output)
    diffusion_output = torch.cat(diffusion_output)
    y = np.stack((y_outs, y_outs_1, y_outs_2, y_outs_3), axis=0)
    
    return targets, dcg_output, diffusion_output, y

# test
if __name__ == '__main__':
    # Add tests here
    logging.info("Successful")
