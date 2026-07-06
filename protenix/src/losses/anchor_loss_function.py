from .abstract_loss_funciton import AbstractLossFunction
import torch

class AnchorLossFunction(AbstractLossFunction):
    def __init__(self, init_s, init_z, pdb_id, lambda_s, lambda_z, device="cuda:0"):
        """
        RMSD loss function for the backbone atoms to reference structure
        """
        self.init_s = init_s
        self.init_z = init_z
        self.pdb_id = pdb_id
        self.device = device
        self.lambda_s = lambda_s
        self.lambda_z = lambda_z

        self.last_adversarial_loss_s = None
        self.last_adversarial_loss_z = None
        self.last_mse_s = None
        self.last_mse_z = None

    def __call__(self, x_0_hat, t, msa=None):
        """
        Usually either fs or gs protein is modeled. Assuming fs is modeled
        """
        if msa is None:
            raise ValueError("MSA is required for anchor loss function")
        optimized_s, optimized_z = msa
        if optimized_z.ndim == 4:
            batch_size = optimized_s.shape[0]
            init_s = self.init_s.unsqueeze(0).expand(batch_size, -1, -1)
            init_z = self.init_z.unsqueeze(0).expand(batch_size, -1, -1, -1)

            per_sample_loss_s = torch.linalg.norm(optimized_s - init_s, dim=(1, 2))
            per_sample_loss_z = torch.linalg.norm((optimized_z - init_z).flatten(1), dim=1)
            adversarial_loss_s = per_sample_loss_s.mean()
            adversarial_loss_z = per_sample_loss_z.mean()

            mse_s = ((optimized_s - init_s) ** 2).mean(dim=(1, 2))
            mse_z = ((optimized_z - init_z) ** 2).mean(dim=(1, 2, 3))

            self.last_adversarial_loss_s = adversarial_loss_s.item()
            self.last_adversarial_loss_z = adversarial_loss_z.item()
            self.last_mse_s = mse_s.mean().item()
            self.last_mse_z = mse_z.mean().item()
        else:
            init_s = self.init_s
            init_z = self.init_z

            adversarial_loss_s = torch.linalg.norm(optimized_s - init_s)
            adversarial_loss_z = torch.linalg.norm(optimized_z - init_z)

            # Last loss function
            self.last_adversarial_loss_s = adversarial_loss_s.item()
            self.last_adversarial_loss_z = adversarial_loss_z.item()
            self.last_mse_s = ((optimized_s - init_s) ** 2).mean().item()
            self.last_mse_z = ((optimized_z - init_z) ** 2).mean().item()

        return (self.lambda_s * adversarial_loss_s, self.lambda_z * adversarial_loss_z), None
    
    def wandb_log(self, x_0_hat):
        return ({"anchor_loss_s": self.last_adversarial_loss_s, "anchor_loss_z": self.last_adversarial_loss_z, "mse_s": self.last_mse_s, "mse_z": self.last_mse_z})