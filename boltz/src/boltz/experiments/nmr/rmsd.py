from __future__ import annotations

# 1-to-1 port of `msa_subsampling/src/protenix/metrics/rmsd.py` (subset used by NMR tooling).

from typing import Optional

import torch


def rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 0.0,
    reduce: bool = True,
):
    assert pred_pose.shape == true_pose.shape  # [..., N, 3]
    if mask is None:
        mask = torch.ones(true_pose.shape[:-1], device=true_pose.device)
    err2 = (torch.square(pred_pose - true_pose).sum(dim=-1) * mask).sum(dim=-1) / mask.sum(dim=-1)
    out = err2.add(eps).sqrt()
    return out.mean() if reduce else out


def align_pred_to_true(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    allowing_reflection: bool = False,
):
    if atom_mask is not None:
        pred_pose = pred_pose * atom_mask.unsqueeze(-1)
        true_pose = true_pose * atom_mask.unsqueeze(-1)
    else:
        atom_mask = torch.ones(*pred_pose.shape[:-1]).to(pred_pose.device)

    if weight is None:
        weight = atom_mask
    else:
        weight = weight * atom_mask

    weighted_n_atoms = torch.sum(weight, dim=-1, keepdim=True).unsqueeze(-1)
    pred_pose_centroid = torch.sum(pred_pose * weight.unsqueeze(-1), dim=-2, keepdim=True) / weighted_n_atoms
    pred_pose_centered = pred_pose - pred_pose_centroid
    true_pose_centroid = torch.sum(true_pose * weight.unsqueeze(-1), dim=-2, keepdim=True) / weighted_n_atoms
    true_pose_centered = true_pose - true_pose_centroid
    H_mat = torch.matmul(
        (pred_pose_centered * weight.unsqueeze(-1)).transpose(-2, -1),
        true_pose_centered * atom_mask.unsqueeze(-1),
    )
    u, _s, v = torch.svd(H_mat)
    u = u.transpose(-1, -2)

    if not allowing_reflection:
        det = torch.linalg.det(torch.matmul(v, u))
        diagonal = torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1)
        rot = torch.matmul(torch.diag_embed(diagonal).to(u.device), u)
        rot = torch.matmul(v, rot)
    else:
        rot = torch.matmul(v, u)

    translate = true_pose_centroid - torch.matmul(pred_pose_centroid, rot.transpose(-1, -2))
    pred_pose_translated = torch.matmul(pred_pose_centered, rot.transpose(-1, -2)) + true_pose_centroid
    return pred_pose_translated, rot, translate


def partially_aligned_rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    align_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    _aligned_pose, rot, translate = align_pred_to_true(
        pred_pose,
        true_pose,
        atom_mask=atom_mask * align_mask,
        weight=weight,
        allowing_reflection=allowing_reflection,
    )
    transformed_pose = torch.matmul(pred_pose, rot.transpose(-1, -2)) + translate
    err_atom = torch.square(transformed_pose - true_pose).sum(dim=-1) * atom_mask
    aligned_mask = atom_mask * align_mask.float()
    unaligned_mask = atom_mask * (1 - align_mask.float())
    aligned_part_err_square = (err_atom * aligned_mask).sum(dim=-1) / aligned_mask.sum(dim=-1)
    unaligned_part_err_square = (err_atom * unaligned_mask).sum(dim=-1) / unaligned_mask.sum(dim=-1)
    aligned_part_rmsd = aligned_part_err_square.add(eps).sqrt()
    unaligned_part_rmsd = unaligned_part_err_square.add(eps).sqrt()
    if reduce:
        aligned_part_rmsd = aligned_part_rmsd.mean()
        unaligned_part_rmsd = unaligned_part_rmsd.mean()
    return aligned_part_rmsd, unaligned_part_rmsd, transformed_pose, rot, translate


def self_aligned_rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    atom_mask: torch.Tensor,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    aligned_rmsd, _, transformed_pred_pose, rot, trans = partially_aligned_rmsd(
        pred_pose=pred_pose,
        true_pose=true_pose,
        align_mask=atom_mask,
        atom_mask=atom_mask,
        eps=eps,
        reduce=reduce,
        allowing_reflection=allowing_reflection,
    )
    return aligned_rmsd, transformed_pred_pose, rot, trans
    pred_pose_centroid = torch.sum(pred_pose * weight.unsqueeze(-1), dim=-2, keepdim=True) / weighted_n_atoms
    pred_pose_centered = pred_pose - pred_pose_centroid
    true_pose_centroid = torch.sum(true_pose * weight.unsqueeze(-1), dim=-2, keepdim=True) / weighted_n_atoms
    true_pose_centered = true_pose - true_pose_centroid
    H_mat = torch.matmul(
        (pred_pose_centered * weight.unsqueeze(-1)).transpose(-2, -1),
        true_pose_centered * atom_mask.unsqueeze(-1),
    )
    u, _s, v = torch.svd(H_mat)
    u = u.transpose(-1, -2)

    if not allowing_reflection:
        det = torch.linalg.det(torch.matmul(v, u))
        diagonal = torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1)
        rot = torch.matmul(torch.diag_embed(diagonal).to(u.device), u)
        rot = torch.matmul(v, rot)
    else:
        rot = torch.matmul(v, u)

    translate = true_pose_centroid - torch.matmul(pred_pose_centroid, rot.transpose(-1, -2))
    pred_pose_translated = torch.matmul(pred_pose_centered, rot.transpose(-1, -2)) + true_pose_centroid
    return pred_pose_translated, rot, translate


def partially_aligned_rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    align_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    _aligned_pose, rot, translate = align_pred_to_true(
        pred_pose,
        true_pose,
        atom_mask=atom_mask * align_mask,
        weight=weight,
        allowing_reflection=allowing_reflection,
    )
    transformed_pose = torch.matmul(pred_pose, rot.transpose(-1, -2)) + translate
    err_atom = torch.square(transformed_pose - true_pose).sum(dim=-1) * atom_mask
    aligned_mask = atom_mask * align_mask.float()
    unaligned_mask = atom_mask * (1 - align_mask.float())
    aligned_part_err_square = (err_atom * aligned_mask).sum(dim=-1) / aligned_mask.sum(dim=-1)
    unaligned_part_err_square = (err_atom * unaligned_mask).sum(dim=-1) / unaligned_mask.sum(dim=-1)
    aligned_part_rmsd = aligned_part_err_square.add(eps).sqrt()
    unaligned_part_rmsd = unaligned_part_err_square.add(eps).sqrt()
    if reduce:
        aligned_part_rmsd = aligned_part_rmsd.mean()
        unaligned_part_rmsd = unaligned_part_rmsd.mean()
    return aligned_part_rmsd, unaligned_part_rmsd, transformed_pose, rot, translate


def self_aligned_rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    atom_mask: torch.Tensor,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    aligned_rmsd, _, transformed_pred_pose, rot, trans = partially_aligned_rmsd(
        pred_pose=pred_pose,
        true_pose=true_pose,
        align_mask=atom_mask,
        atom_mask=atom_mask,
        eps=eps,
        reduce=reduce,
        allowing_reflection=allowing_reflection,
    )
    return aligned_rmsd, transformed_pred_pose, rot, trans

