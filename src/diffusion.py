"""
Build basic structure of diffusion pipeline
This script assumes a fully function DCG module and Dataloader module
"""
import torch
import numpy as np

class DiffusionBaseUtils():
    def __init__(self, timesteps = 1000, noise_schedule = "Linear"):
        super(DiffusionBaseUtils,self).__init__()
        self.T = timesteps # Number of diffusion steps
        self.noise_schedule = noise_schedule
        # add multivariate gaussian attribute

    def get_noise_schedule(self, beta_initial = 0.0001, beta_final = 0.02):
        """
        Returns a noise schedule for the forward diffusion process. Noise schedule is an increasing sequence
        as in diffusion we generally start by snall perturbations and then keep on increasing the perturbations 
        (i.e. noise scale, as images keep getting noisier).
        """
        if self.noise_schedule == "Linear":
            betas = torch.linspace(beta_initial, beta_final, self.T)  
        else:
            raise NotImplementedError
        return betas
    
    def get_alpha_prod(self, timestep = None):
        """
        Returns alpha_prod which is the product of alpha_t where
        alpha_t = 1 - beta_t for all time until timestep
        """
        if timestep == None:
            timestep = self.T
        alphas = 1-self.get_noise_schedule()[:timestep]
        alpha_prod = torch.prod(alphas)
        return alpha_prod
    
class ForwardDiffusionUtils(DiffusionBaseUtils):
    def __init__(self):
        super(ForwardDiffusionUtils,self).__init__()

    def forward_diffusion(self, var, noising_condition, t):
        """
        This method is used to add noise to y_0 (whatever that is), global_prior and local prior and then 
        obtain the respective noisy variables following equation 2 of paper.
        y_0, global and local priors will be obtained form dcg.

        t is the timestep till which nosie has been added in the forward diffusion process
        var is the variable on which we are adding noise

        Note:
        * noising_condition = (global_prior + local_prior)/2 for y_0
        * noising_condition = global_prior for y_global
        * noising_condition = local_prior for y_local
        """
        eps = torch.randn_like(var) # gaussian noise
        alpha_prod = self.get_alpha_prod(timestep=t) # generate alpha_prod for t time (where t is time for which noise has been added)
        noised_var = torch.sqrt(alpha_prod)*var + torch.sqrt(1-alpha_prod)*eps + (1-torch.sqrt(alpha_prod))*noising_condition # add noise till timestep = T

        return noised_var
    


class ReverseDiffusionUtils(DiffusionBaseUtils):
    def __init__(self):
        super(ReverseDiffusionUtils,self).__init__()
        """
        NOTES: - Sehmimul

        1. The diffusion pipeline in DiffMIC is based off https://arxiv.org/abs/2206.07275 
        Github: https://github.com/XzwHan/CARD/blob/ebe2f2ae95dc7a7a95e6a71c0c8e1cabf8451087/classification/diffusion_utils.py#L106
        Especially the CARD paper's Algorithm 2 line 4 is very useful as thta talks about inference and
        the same inference scheme has been applied in the DiffMIC paper
        Thus CARD has to be an intergal part of our work.

        2. I understand p_sample, but I don't understand the goal of p_sample_t_1to0. The goal of y_0_reparam is outlined in point 1
        """

    def reverse_diffusion(self):
        """
        This is similar to p_sample of code
        and 
        """
        raise NotImplementedError


# test
if __name__ == '__main__':
    timesteps = 3
    df=DiffusionBaseUtils(timesteps = timesteps)

    # ================================= Tests ===========================================
    ## testing utils for forward diffusion method in the 3 lines below
    # fd = ForwardDiffusionUtils()
    # noised_var = fd.forward_diffusion(torch.tensor([1,2], dtype=torch.float32), 0, 1000)
    # print(noised_var)