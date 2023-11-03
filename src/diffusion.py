"""
Build basic structure of diffusion pipeline
This script assumes a fully function DCG module and Dataloader module
"""
import torch
import numpy as np

class DiffusionBaseUtils():
    def __init__(self, timesteps = 1000, noise_schedule = "Linear", beta_initial = 0.0001, beta_final = 0.02):
        super(DiffusionBaseUtils,self).__init__()
        self.T = timesteps # Number of diffusion steps
        self.noise_schedule = noise_schedule
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        # add multivariate gaussian attribute

    @property
    def noise_list(self):
        """
        Returns a noise schedule for the forward diffusion process. Noise schedule is an increasing sequence
        as in diffusion we generally start by snall perturbations and then keep on increasing the perturbations 
        (i.e. noise scale, as images keep getting noisier).
        """
        if self.noise_schedule == "Linear":
            betas = torch.linspace(self.beta_initial, self.beta_final, self.T)  
        else:
            raise NotImplementedError("{0} noise schedule not implemented".format(self.noise_schedule))
        return betas
    
    def get_alpha_prod(self, timestep = None):
        """
        Returns alpha_prod which is the product of alpha_t where
        alpha_t = 1 - beta_t for all time until timestep
        """
        if timestep == None:
            timestep = self.T
        alphas = 1-self.noise_list[:timestep]
        alpha_prod = torch.prod(alphas)
        return alpha_prod
    
class ForwardDiffusionUtils(DiffusionBaseUtils):
    def __init__(self):
        super(ForwardDiffusionUtils,self).__init__()

    def forward_diffusion(self, var, noising_condition, t): # rename noising condition as that is not expressive
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
        Thus CARD has to be an intergal part of our work. Output of p_sample is literally the denoising
        process of the CARD paper.

        2. I understand p_sample, but I don't understand the goal of p_sample_t_1to0. The goal of y_0_reparam is outlined in point 1

        3.  p_sample is just reverse process of CARD paper and it is just equation 9 and Algorithm 2 and it

        4. extract function is not necessary and the x or y input is only required to get the 'shape' and it is not really useful otherwise

        5. In p_sample_loop curl_y is just a sample drawn from the N(prior, I) distribution.
        """
    def reverse_diffusion_parameters(self, t):
        """
        The parameters returned with this helper function are used in the reverse diffusion process of the 
        CARD paper
        """
        beta_t = self.noise_list[t] # getting beta_t at timestep t
        alpha_prod_t = self.get_alpha_prod(timestep=t)

        if t<1:
            raise ValueError("time step 0 error. Please fix it")
        alpha_prod_t_m1 = self.get_alpha_prod(timestep=t-1)  ### will throw error at time step t = 0 MAKE SURE TO DEAL WITH IT

        gamma_0 = beta_t*(torch.sqrt(alpha_prod_t_m1)/(1 - alpha_prod_t))
        gamma_1 = ((1-alpha_prod_t_m1)*torch.sqrt(1-beta_t))/(1-alpha_prod_t)
        gamma_2 = 1 + ((torch.sqrt(alpha_prod_t)-1)*(torch.sqrt(1-beta_t)+torch.sqrt(alpha_prod_t_m1)))/(1-alpha_prod_t)
        beta_var = ((1-alpha_prod_t_m1)*beta_t)/(1-alpha_prod_t)

        return gamma_0, gamma_1, gamma_2, beta_var, alpha_prod_t


    def reverse_diffusion_step(self, x, y_t, t, cond_prior, score_net): # cannot test without cond_prior and score_net
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
        gamma_0, gamma_1, gamma_2, beta_var, alpha_prod_t = self.reverse_diffusion_parameters(t = t)
        eps = torch.randn_like(y_t)


        # first we reparameterize y0 to obtain y0_hat
        y0_hat = (1/torch.sqrt(alpha_prod_t))*(y_t - (1-torch.sqrt(alpha_prod_t)*cond_prior - torch.sqrt(1-alpha_prod_t)*score_net(x,y_t,cond_prior,t)))

        y_tm1 = gamma_0*y0_hat+gamma_1*y_t+gamma_2*cond_prior+torch.sqrt(beta_var)*eps

        return y_tm1, y0_hat
    
    def full_reverse_diffusion(self, x, cond_prior, score_net): # cannot test without cond_prior and score_net
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
            ### *** POTENTIAL_BUG ****
            t = self.T - t - 1
            if t > 0:
                y_tm1, y0_hat = self.reverse_diffusion_step(self, x, y_t, t, cond_prior, score_net)
                y_t = y_tm1
            else:
                # so t = 1
                y_tm1 = y0_hat
        y0_synthetic = y_tm1
        return y0_synthetic


# test
if __name__ == '__main__':
    # timesteps = 3
    # df=DiffusionBaseUtils(timesteps = timesteps)
    rd = ReverseDiffusionUtils()
    print(rd.reverse_diffusion_parameters(t=5))

    # ================================= Tests ===========================================
    ## testing utils for forward diffusion method in the 3 lines below
    # fd = ForwardDiffusionUtils()
    # noised_var = fd.forward_diffusion(torch.tensor([1,2], dtype=torch.float32), 0, 1000)
    # print(noised_var)