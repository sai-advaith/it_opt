"""Differentiable geometric restraints for IT-opt.

All restraints act on a batch of denoised structures x0_hat of shape
[B, N_atom, 3] (Angstrom) and return a per-sample violation that is turned
into a flat-bottom squared-hinge loss.  They are framework agnostic: the only
thing they need is a mapping from (chain, seq_pos, atom_name) to the atom
index of the model's atom array, which `AtomIndexer` provides.

Restraint types (JSON "type" field):

distance      flat-bottom bound on |x_a - x_b| (each side may be an atom group;
              the group centroid is used).
position      after superposing `frame` atoms onto their reference coordinates,
              each restrained atom (or the centroid of the group when one target
              is given) must lie within `tolerance` of its target point.
substructure  Kabsch-superposed RMSD of an atom set to reference coordinates
              must be <= `tolerance`.
orientation   rigid pose of domain2 relative to domain1 must match the
              reference relative pose: rotation angle error <= `angle_tol`
              (deg) and translation error <= `dist_tol` (A).

Kabsch superposition is done through a 3x3 SVD on CPU in float64 (tiny,
and MPS lacks a stable SVD); gradients flow back to the device tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# atom lookup
# --------------------------------------------------------------------------- #
class AtomIndexer:
    """Maps (chain_label, seq_pos (1-based), atom_name) -> atom-array index.

    `chain_labels` gives the user-facing chain id for each chain of the model's
    atom array, in the order the chains appear there.
    """

    def __init__(self, atom_array, chain_labels):
        model_chains = list(dict.fromkeys(atom_array.chain_id.tolist()))
        if len(model_chains) != len(chain_labels):
            raise ValueError(
                f"model has {len(model_chains)} chains {model_chains}, spec has {len(chain_labels)}"
            )
        self.model_to_label = dict(zip(model_chains, chain_labels))
        self.index = {}
        for i, (c, r, a) in enumerate(
            zip(atom_array.chain_id, atom_array.res_id, atom_array.atom_name)
        ):
            self.index[(self.model_to_label[c], int(r), str(a))] = i

    def __call__(self, atoms):
        out = []
        for c, r, a in atoms:
            key = (str(c), int(r), str(a))
            if key not in self.index:
                raise KeyError(f"atom {c}:{r}@{a} not present in the predicted model")
            out.append(self.index[key])
        return out


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def kabsch(mobile: torch.Tensor, target: torch.Tensor):
    """Optimal rotation R and translation t with R @ mobile + t ~= target.

    mobile: [B, n, 3], target: [n, 3] or [B, n, 3].  Returns R [B,3,3], t [B,3].
    """
    if target.dim() == 2:
        target = target.unsqueeze(0).expand_as(mobile)
    mc = mobile.mean(dim=1, keepdim=True)
    tc = target.mean(dim=1, keepdim=True)
    H = (mobile - mc).transpose(1, 2) @ (target - tc)  # [B,3,3]
    Hc = H.cpu().double()
    U, S, Vh = torch.linalg.svd(Hc)
    d = torch.sign(torch.det(Vh.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    R = (Vh.transpose(1, 2) @ D @ U.transpose(1, 2)).to(mobile.dtype).to(mobile.device)
    t = tc.squeeze(1) - (R @ mc.squeeze(1).unsqueeze(-1)).squeeze(-1)
    return R, t


def apply(R, t, x):
    return (R @ x.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)


def rotation_angle(R: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (radians) of rotation matrices [B,3,3], safe gradient."""
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    c = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(c)


def flat_bottom(value, lower=None, upper=None):
    loss = torch.zeros_like(value)
    if upper is not None:
        loss = loss + torch.relu(value - upper) ** 2
    if lower is not None:
        loss = loss + torch.relu(lower - value) ** 2
    return loss


# --------------------------------------------------------------------------- #
# restraint classes
# --------------------------------------------------------------------------- #
@dataclass
class Restraint:
    id: str
    weight: float = 1.0
    label: str = ""

    def measure(self, x):  # -> (value [B], loss [B])
        raise NotImplementedError

    def describe(self, value):
        return f"{value:.2f}"


class DistanceRestraint(Restraint):
    def __init__(self, spec, idx: AtomIndexer, device):
        super().__init__(spec["id"], float(spec.get("weight", 1.0)), spec.get("label", ""))
        self.ia = torch.tensor(idx(spec["a"]), device=device)
        self.ib = torch.tensor(idx(spec["b"]), device=device)
        self.lower = spec.get("lower")
        self.upper = spec.get("upper")
        self.unit = "A"

    def measure(self, x):
        pa = x[:, self.ia].mean(1)
        pb = x[:, self.ib].mean(1)
        d = (pa - pb).norm(dim=-1)
        return d, flat_bottom(d, self.lower, self.upper)


class PositionRestraint(Restraint):
    def __init__(self, spec, idx: AtomIndexer, device):
        super().__init__(spec["id"], float(spec.get("weight", 1.0)), spec.get("label", ""))
        self.iatoms = torch.tensor(idx(spec["atoms"]), device=device)
        self.targets = torch.tensor(spec["targets"], dtype=torch.float32, device=device)
        self.iframe = torch.tensor(idx(spec["frame"]["atoms"]), device=device)
        self.frame_ref = torch.tensor(spec["frame"]["coords"], dtype=torch.float32, device=device)
        self.tol = float(spec.get("tolerance", 1.0))
        self.centroid = self.targets.shape[0] == 1 and len(self.iatoms) > 1
        if not self.centroid and self.targets.shape[0] != len(self.iatoms):
            raise ValueError(f"{self.id}: {len(self.iatoms)} atoms but {self.targets.shape[0]} targets")
        self.unit = "A"

    def measure(self, x):
        R, t = kabsch(x[:, self.iframe], self.frame_ref)
        p = apply(R, t, x[:, self.iatoms])
        if self.centroid:
            dev = (p.mean(1) - self.targets[0]).norm(dim=-1)  # [B]
            return dev, flat_bottom(dev, None, self.tol)
        dev = (p - self.targets.unsqueeze(0)).norm(dim=-1)  # [B, n]
        # report RMS deviation, penalise each atom separately
        rms = dev.pow(2).mean(1).sqrt()
        return rms, flat_bottom(dev, None, self.tol).mean(1)


class SubstructureRestraint(Restraint):
    def __init__(self, spec, idx: AtomIndexer, device):
        super().__init__(spec["id"], float(spec.get("weight", 1.0)), spec.get("label", ""))
        self.iatoms = torch.tensor(idx(spec["atoms"]), device=device)
        self.ref = torch.tensor(spec["coords"], dtype=torch.float32, device=device)
        self.tol = float(spec.get("tolerance", 1.0))
        self.unit = "A RMSD"

    def measure(self, x):
        mob = x[:, self.iatoms]
        R, t = kabsch(mob, self.ref)
        rmsd = (apply(R, t, mob) - self.ref).pow(2).sum(-1).mean(-1).add(1e-8).sqrt()
        return rmsd, flat_bottom(rmsd, None, self.tol)


class OrientationRestraint(Restraint):
    """Relative rigid pose of two domains.

    Reference: domain coords D1, D2 (in one common frame).  For a prediction,
    superpose domain1 onto D1 -> brings prediction into the reference frame.
    Then superpose the (transformed) predicted domain2 onto D2; the residual
    rigid transform (R_err, t_err) is the error in the relative pose.
    """

    def __init__(self, spec, idx: AtomIndexer, device):
        super().__init__(spec["id"], float(spec.get("weight", 1.0)), spec.get("label", ""))
        self.i1 = torch.tensor(idx(spec["domain1"]["atoms"]), device=device)
        self.i2 = torch.tensor(idx(spec["domain2"]["atoms"]), device=device)
        self.D1 = torch.tensor(spec["domain1"]["coords"], dtype=torch.float32, device=device)
        self.D2 = torch.tensor(spec["domain2"]["coords"], dtype=torch.float32, device=device)
        self.angle_tol = math.radians(float(spec.get("angle_tol", 10.0)))
        self.dist_tol = float(spec.get("dist_tol", 2.0))
        # 1 rad of angular error is weighted like `angle_scale` A of translation
        self.angle_scale = float(spec.get("angle_scale", 10.0))
        self.unit = "deg"

    def measure(self, x):
        R1, t1 = kabsch(x[:, self.i1], self.D1)
        p2 = apply(R1, t1, x[:, self.i2])  # predicted domain 2 in reference frame
        R2, _ = kabsch(p2, self.D2)
        ang = rotation_angle(R2)
        shift = (p2.mean(1) - self.D2.mean(0)).norm(dim=-1)
        loss = (self.angle_scale ** 2) * flat_bottom(ang, None, self.angle_tol) + flat_bottom(
            shift, None, self.dist_tol
        )
        self._last_shift = shift.detach()
        return torch.rad2deg(ang), loss

    def describe(self, value):
        return f"{value:.1f} deg"


RESTRAINT_TYPES = {
    "distance": DistanceRestraint,
    "position": PositionRestraint,
    "substructure": SubstructureRestraint,
    "orientation": OrientationRestraint,
}


# --------------------------------------------------------------------------- #
# loss function compatible with the it_opt AbstractLossFunction interface
# --------------------------------------------------------------------------- #
class GeometricRestraintLoss:
    def __init__(self, restraint_specs, atom_indexer: AtomIndexer, device, scale=1.0):
        self.restraints = [RESTRAINT_TYPES[s["type"]](s, atom_indexer, device) for s in restraint_specs]
        self.scale = scale
        self.last = None

    def __call__(self, x0_hat, time=None, msa=None):
        total = x0_hat.new_zeros(())
        report = []
        for r in self.restraints:
            value, loss = r.measure(x0_hat)
            total = total + r.weight * loss.mean()
            report.append(
                {
                    "id": r.id,
                    "value_mean": float(value.detach().mean()),
                    "values": [float(v) for v in value.detach().cpu()],
                    "loss": float(loss.detach().mean()),
                    "satisfied": [bool(l < 1e-6) for l in loss.detach().cpu()],
                }
            )
        self.last = report
        return self.scale * total, report

    def wandb_log(self, x0_hat):
        return {}
