import torch
from biotite.structure import connect_via_residue_names
import numpy as np
import gemmi

class AbstractLossFunction:  
    def wandb_log(self, x_0_hat):
        pass

    def __call__(self, x_0_hat, time, msa=None):
        pass
    
    def post_optimization_step(self):
        pass

    def save_state(self, structures, folder_path):
        pass