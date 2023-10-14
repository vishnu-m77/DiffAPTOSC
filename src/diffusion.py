"""
Build basic structure of diffusion pipeline
This script assumes a fully function DCG module and Dataloader module
"""
import torch

class DiffusionBaseUtils():
    def __init__(self, timesteps = 1000, noise_schedule = "Linear"):
        super(DiffusionBaseUtils,self).__init__()
        self.T = timesteps # Number of diffusion steps
        self.noise_schedule = noise_schedule

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
    
    def get_alpha_prod(self):
        """
        Use t to get alpha at timestep t
        """
        raise NotImplementedError
    
class ForwardDiffusionUtils():
    def __init__(self):
        super(ForwardDiffusionUtils,self).__init__()
    


class ForwardDiffusionUtils():
    def __init__(self):
        super(ForwardDiffusionUtils,self).__init__()


# test
if __name__ == '__main__':
    df=DiffusionBaseUtils()
    # print(df.get_noise_schedule())