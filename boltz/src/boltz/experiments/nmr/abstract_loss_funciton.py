"""Abstract loss base class for the vendored NMR/loss stack.

This file is intentionally minimal and should not import any repo-specific
utilities. Individual loss implementations (e.g. `violation_loss_function.py`)
should import what they need.
"""

class AbstractLossFunction:
    def wandb_log(self, x_0_hat):
        pass

    def __call__(self, x_0_hat, time, msa=None):
        pass
    
    def post_optimization_step(self):
        pass

    def save_state(self, structures, folder_path):
        pass