from .abstract_loss_funciton import AbstractLossFunction
from .openfold_violations.violations import find_structural_violations, get_atom14_positions


class ViolationLossFunction(AbstractLossFunction):
    def __init__(self, atom_array, lambda_violation):
        self.atom_array = atom_array
        self.lambda_violation = lambda_violation
        self._last_loss = None

    def get_violations_loss(self, x_0_hat):
        x_0_hat_14atoms = get_atom14_positions(self.atom_array, x_0_hat)
        violations_dict = find_structural_violations(x_0_hat_14atoms, self.atom_array)
        violation_loss = violations_dict["between_residues"]["clashes_mean_loss"] + violations_dict["between_residues"]["connections_per_residue_loss_sum"].mean() + violations_dict["within_residues"]["per_atom_loss_sum"].mean()
        return violation_loss
    
    def get_weighted_violations_loss(self, x_0_hat, weights=None):
        if weights is None:
            violation_loss = self.get_violations_loss(x_0_hat)
            self._last_loss = violation_loss.item()
            return violation_loss * self.lambda_violation, None

        weighted_violation_loss = 0
        for i in range(x_0_hat.shape[0]):
            viol_loss_i = self.get_violations_loss(x_0_hat[i][None])
            weighted_violation_loss = weighted_violation_loss + viol_loss_i * weights[i]
        self._last_loss = self.lambda_violation * weighted_violation_loss.item()
        return weighted_violation_loss, None

    def __call__(self, x_0_hat, time, msa=None):
        loss = self.get_violations_loss(x_0_hat)
        self._last_loss = loss.item()
        return loss * self.lambda_violation, None
    
    def wandb_log(self, x_0_hat):
        return {"violation loss": self._last_loss}