# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Embedding optimization for AlphaFold 3.

Optimizes trunk embeddings (single/pair) at each diffusion denoising step
to maximize confidence scores (iPTM/PTM/pLDDT). The diffusion loop is unrolled
and embeddings are updated per-step, with the updated embeddings passed to the
next denoising iteration.

Architecture:
  for outer_step in range(outer_diffusion_steps):
    positions = initial_noise
    for diff_step in range(diffusion_steps):
      positions_noisy = augment_and_noise(positions, ...)
      if diff_step % optimization_interval == 0:
        for opt_step in range(optimization_steps):
          grads, denoised = grad_fn(embeddings, positions_noisy, noise_level)
          embeddings = adam_update(embeddings, grads)
      else:
        denoised = denoise(embeddings, positions_noisy, noise_level)
      positions = euler_update(positions_noisy, denoised, ...)
"""

import csv
import dataclasses
import functools
import os
import re
from typing import Any

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp


# From diffusion_head.py
_SIGMA_DATA = 16.0


def soft_dgram_from_positions(positions, min_bin=3.25, max_bin=50.75, num_bins=39,
                               temperature=1.0):
  """Compute soft distogram from positions with differentiable binning.

  Uses sigmoid functions instead of hard comparisons to allow gradient flow.
  This is used during optimization to enable gradients through the confidence
  head's distogram features.

  Args:
    positions: (num_res, 3) Position coordinates.
    min_bin: Minimum distance bin.
    max_bin: Maximum distance bin.
    num_bins: Number of bins.
    temperature: Temperature for sigmoid (lower = sharper, more like hard bins).

  Returns:
    Soft distogram with shape (num_res, num_res, num_bins).
  """
  lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
  lower_breaks_sq = jnp.square(lower_breaks)
  upper_breaks_sq = jnp.concatenate(
      [lower_breaks_sq[1:], jnp.array([1e8], dtype=jnp.float32)], axis=-1
  )

  # Compute squared distances
  dist2 = jnp.sum(
      jnp.square(
          jnp.expand_dims(positions, axis=-2)
          - jnp.expand_dims(positions, axis=-3)
      ),
      axis=-1,
      keepdims=True,
  )

  # Soft binning using sigmoid instead of hard comparisons
  # sigmoid((dist2 - lower) / temp) ≈ 1 when dist2 > lower
  # sigmoid((upper - dist2) / temp) ≈ 1 when dist2 < upper
  soft_lower = jax.nn.sigmoid((dist2 - lower_breaks_sq) / temperature)
  soft_upper = jax.nn.sigmoid((upper_breaks_sq - dist2) / temperature)

  dgram = soft_lower * soft_upper
  return dgram


def soft_dgram_from_positions_config(positions, config, temperature=1.0):
  """Soft distogram matching the signature of template_modules.dgram_from_positions."""
  return soft_dgram_from_positions(
      positions,
      min_bin=config.min_bin,
      max_bin=config.max_bin,
      num_bins=config.num_bins,
      temperature=temperature,
  )


@dataclasses.dataclass(frozen=True)
class EmbeddingOptimizationConfig:
  """Configuration for per-step embedding optimization during diffusion."""

  enabled: bool = False
  # Number of full diffusion passes (outer loop).
  outer_steps: int = 1
  # Number of gradient steps per diffusion step (inner loop).
  optimization_steps: int = 1
  # Run optimization every N diffusion steps (1 = every step).
  optimization_interval: int = 1
  # Number of diffusion denoising steps during optimization.
  # Can be fewer than the full 200 for speed. Set to 0 to use the model's
  # default eval steps.
  diffusion_steps: int = 0
  learning_rate: float = 1e-4
  max_grad_norm: float = 1.0
  anchor_lambda_single: float = 0.01
  anchor_lambda_pair: float = 0.01
  # Whether to optimize single embeddings. FKSFold-Boltz freezes single
  # (s_trunk) by default and only optimizes pair (z_trunk).
  optimize_single: bool = False
  scoring_type: str = 'iptm'
  # Structured update mode for pair embeddings.
  # Options: 'full', 'low_rank', 'per_residue_bias', 'channel_scale',
  #          'low_rank_diagonal', 'element_scale', 'element_scale_channel',
  #          'element_scale_residue', 'element_scale_residue_channel'
  update_mode: str = 'full'
  # Rank for low-rank update modes ('low_rank' and 'low_rank_diagonal').
  low_rank: int = 4
  # ipSAE parameters
  ipsae_pae_cutoff: float = 10.0  # PAE cutoff in Angstroms
  ipsae_soft_temperature: float = 1.0  # Temperature for soft PAE mask
  # Number of coordinate samples to average over when computing confidence loss.
  # Higher values give more robust gradient estimates but use more memory.
  optimization_batch_size: int = 1
  # If True, compute gradients for each sample sequentially and average
  # (gradient accumulation). Uses the same memory as batch_size=1 regardless
  # of the actual batch size, at the cost of batch_size sequential forward/
  # backward passes instead of one parallel vmap pass.
  gradient_accumulation: bool = False
  # Diffusion sampling parameters (overrides model defaults during optimization)
  gamma_0: float = 0.8
  gamma_min: float = 1.0
  noise_scale: float = 1.003
  step_scale: float = 1.5
  # Perturbation cliff: maximum allowed perturbation as % of original norm.
  # When the perturbation reaches this threshold, optimization stops and
  # normal diffusion generation continues. Set to 0.0 to disable.
  perturbation_cliff_pct: float = 0.0
  # Score threshold: stop optimization when the score (defined by scoring_type)
  # reaches this value. Set to 1.0 to disable (scores are always <= 1.0).
  score_threshold: float = 1.0
  # If True, only update cross-chain (inter-chain) pair embeddings.
  # Intra-chain pair embeddings remain frozen at their initial values.
  # This focuses optimization on the protein-protein interface.
  cross_chain_only: bool = False
  # Pairformer projection: optional projection step to apply Pairformer
  # to the updated embeddings. This is applied with stop_gradient (no backprop)
  # to project embeddings back towards the learned manifold.
  pairformer_projection_enabled: bool = False
  # Apply Pairformer projection every N total embedding updates.
  # Counts across all outer/diffusion/optimization steps (not per-inner-loop).
  # Set to 1 to apply after every update, 5 to apply every 5 updates, etc.
  pairformer_projection_interval: int = 1


_VALID_UPDATE_MODES = (
    'full', 'low_rank', 'per_residue_bias', 'channel_scale',
    'low_rank_diagonal', 'element_scale', 'element_scale_channel',
    'element_scale_residue', 'element_scale_residue_channel',
    'zero_sum', 'zero_sum_channel',
)


def init_update_params(
    embeddings_init: dict[str, jnp.ndarray],
    config: EmbeddingOptimizationConfig,
    cross_chain_mask: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
  """Initialize structured optimization parameters.

  For 'full' mode, the parameters are the raw embeddings themselves.
  For structured modes, the parameters are compact representations
  (e.g., low-rank factors) initialized to zero so that the initial
  reconstruction equals embeddings_init.

  Args:
    embeddings_init: Initial embeddings with 'single' and 'pair' keys.
    config: Optimization config specifying update_mode and low_rank.
    cross_chain_mask: Optional boolean mask of shape [N, N] where True indicates
      cross-chain (inter-chain) pairs. If provided and config.cross_chain_only
      is True, only cross-chain pairs will be updated during optimization.

  Returns:
    Dictionary of optimization parameters (leaves for jax.tree operations).
  """
  mode = config.update_mode
  if mode not in _VALID_UPDATE_MODES:
    raise ValueError(
        f'Unknown update_mode: {mode}. Must be one of {_VALID_UPDATE_MODES}.'
    )

  pair_init = embeddings_init['pair']  # (N, N, C)
  n_tokens, _, pair_channels = pair_init.shape
  rank = config.low_rank

  params = {}

  # Single embedding params (full mode always, only if optimizing)
  if config.optimize_single:
    params['single'] = embeddings_init['single'].copy()

  # Pair embedding params (structured)
  if mode == 'full':
    params['pair'] = pair_init.copy()
  elif mode == 'low_rank':
    # pair = pair_init + einsum('irc,jrc->ijc', U, V)
    # Initialize to zero so initial reconstruction = pair_init.
    params['pair_U'] = jnp.zeros((n_tokens, rank, pair_channels))
    params['pair_V'] = jnp.zeros((n_tokens, rank, pair_channels))
  elif mode == 'per_residue_bias':
    # pair = pair_init + left[:, None, :] + right[None, :, :]
    params['pair_left'] = jnp.zeros((n_tokens, pair_channels))
    params['pair_right'] = jnp.zeros((n_tokens, pair_channels))
  elif mode == 'channel_scale':
    # pair = alpha * pair_init + beta
    params['pair_alpha'] = jnp.ones((pair_channels,))
    params['pair_beta'] = jnp.zeros((pair_channels,))
  elif mode == 'low_rank_diagonal':
    # pair = pair_init + einsum('irc,jrc->ijc', U, V) + eye * diag
    params['pair_U'] = jnp.zeros((n_tokens, rank, pair_channels))
    params['pair_V'] = jnp.zeros((n_tokens, rank, pair_channels))
    params['pair_diag'] = jnp.zeros((n_tokens, pair_channels))
  elif mode == 'element_scale':
    # pair = (1 + beta) * pair_init, uniform multiplicative scaling.
    # beta is a single scalar initialized to zero so initial z = z_init.
    params['pair_beta'] = jnp.zeros(())
  elif mode == 'element_scale_channel':
    # pair = (1 + beta[None, None, :]) * pair_init, per-channel scaling.
    # beta has shape [C], one scale factor per channel.
    params['pair_beta'] = jnp.zeros((pair_channels,))
  elif mode == 'element_scale_residue':
    # pair = (1 + beta[:, None] * beta[None, :])[:,:,None] * pair_init
    # beta has shape [N], applied as rank-1 outer product over residue pairs.
    # Initialize to small non-zero to avoid zero gradient at initialization
    # (outer product has d(beta_i*beta_j)/d(beta_i) = beta_j which is 0 when beta=0)
    params['pair_beta'] = jnp.ones((n_tokens,)) * 0.01
  elif mode == 'element_scale_residue_channel':
    # pair = (1 + einsum('ic,jc->ijc', beta, beta)) * pair_init
    # beta has shape [N, C], per-channel rank-1 outer product.
    # Initialize to small non-zero to avoid zero gradient at initialization
    params['pair_beta'] = jnp.ones((n_tokens, pair_channels)) * 0.01
  elif mode == 'zero_sum':
    # pair = pair_init + (A - A.mean(axis=(0,1), keepdim=True))
    # The perturbation Delta = A - mean(A) has zero sum over (i,j) for each
    # channel, so pair.sum(axis=(0,1)) = pair_init.sum(axis=(0,1)) is preserved.
    # Initialize A to zero so initial reconstruction = pair_init.
    params['pair_A'] = jnp.zeros_like(pair_init)
  elif mode == 'zero_sum_channel':
    # pair = pair_init + (A - A.mean(axis=-1, keepdims=True))
    # The perturbation Delta = A - mean(A, axis=-1) has zero sum over channels
    # for each (i,j) position, so pair.sum(axis=-1) = pair_init.sum(axis=-1)
    # is preserved. This keeps the (N, N) spatial structure when summed over C.
    # Initialize A to zero so initial reconstruction = pair_init.
    params['pair_A'] = jnp.zeros_like(pair_init)

  # Store cross-chain mask if provided (not optimized, just stored for apply_update)
  if config.cross_chain_only and cross_chain_mask is not None:
    # Store as float for multiplication, with stop_gradient to prevent optimization
    params['_cross_chain_mask'] = jax.lax.stop_gradient(
        cross_chain_mask.astype(jnp.float32)
    )

  return params


def apply_update(
    params: dict[str, jnp.ndarray],
    embeddings_init: dict[str, jnp.ndarray],
    config: EmbeddingOptimizationConfig,
) -> dict[str, jnp.ndarray]:
  """Reconstruct full embeddings from structured parameters.

  Args:
    params: Optimization parameters from init_update_params.
    embeddings_init: Frozen initial embeddings.
    config: Optimization config.

  Returns:
    Full embeddings dict with 'single' and 'pair' keys.
  """
  mode = config.update_mode
  pair_init = embeddings_init['pair']

  # Reconstruct single
  if config.optimize_single:
    single = params['single']
  else:
    single = embeddings_init['single']

  # Reconstruct pair
  if mode == 'full':
    pair = params['pair']
  elif mode == 'low_rank':
    pair = pair_init + jnp.einsum(
        'irc,jrc->ijc', params['pair_U'], params['pair_V']
    )
  elif mode == 'per_residue_bias':
    pair = (
        pair_init
        + params['pair_left'][:, None, :]
        + params['pair_right'][None, :, :]
    )
  elif mode == 'channel_scale':
    pair = (
        params['pair_alpha'][None, None, :] * pair_init
        + params['pair_beta'][None, None, :]
    )
  elif mode == 'low_rank_diagonal':
    low_rank = jnp.einsum(
        'irc,jrc->ijc', params['pair_U'], params['pair_V']
    )
    n = pair_init.shape[0]
    diag_update = jnp.eye(n)[:, :, None] * params['pair_diag'][:, None, :]
    pair = pair_init + low_rank + diag_update
  elif mode == 'element_scale':
    pair = (1 + params['pair_beta']) * pair_init
  elif mode == 'element_scale_channel':
    pair = (1 + params['pair_beta'][None, None, :]) * pair_init
  elif mode == 'element_scale_residue':
    beta = params['pair_beta']
    scale = 1 + beta[:, None] * beta[None, :]  # [N, N]
    pair = scale[:, :, None] * pair_init
  elif mode == 'element_scale_residue_channel':
    beta = params['pair_beta']  # [N, C]
    scale = 1 + jnp.einsum('ic,jc->ijc', beta, beta)  # [N, N, C]
    pair = scale * pair_init
  elif mode == 'zero_sum':
    # pair = pair_init + (A - A.mean(axis=(0,1)))
    # Delta has zero sum over (i,j) for each channel, preserving marginal sums.
    A = params['pair_A']
    delta = A - jnp.mean(A, axis=(0, 1), keepdims=True)
    pair = pair_init + delta
  elif mode == 'zero_sum_channel':
    # pair = pair_init + (A - A.mean(axis=-1))
    # Delta has zero sum over channels for each (i,j), preserving the (N,N)
    # spatial structure when summed over channels.
    A = params['pair_A']
    delta = A - jnp.mean(A, axis=-1, keepdims=True)
    pair = pair_init + delta
  else:
    raise ValueError(f'Unknown update_mode: {mode}')

  # Apply cross-chain mask if enabled: only update inter-chain pairs
  if config.cross_chain_only and '_cross_chain_mask' in params:
    cross_chain_mask = params['_cross_chain_mask']  # [N, N] float
    delta = pair - pair_init
    pair = pair_init + delta * cross_chain_mask[:, :, None]

  return {'single': single, 'pair': pair}


def _noise_schedule(t, smin=0.0004, smax=160.0, p=7):
  """Noise schedule matching diffusion_head.noise_schedule."""
  return (
      _SIGMA_DATA
      * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
  )


def _random_rotation(key):
  """Random rotation via Gram-Schmidt."""
  v0, v1 = jax.random.normal(key, shape=(2, 3))
  e0 = v0 / jnp.maximum(1e-10, jnp.linalg.norm(v0))
  v1 = v1 - e0 * jnp.dot(v1, e0, precision=jax.lax.Precision.HIGHEST)
  e1 = v1 / jnp.maximum(1e-10, jnp.linalg.norm(v1))
  e2 = jnp.cross(e0, e1)
  return jnp.stack([e0, e1, e2])


def _random_augmentation(rng_key, positions, mask):
  """Apply random rigid augmentation to atom positions."""
  from alphafold3.model.components import utils  # pylint: disable=g-import-not-at-top
  rotation_key, translation_key = jax.random.split(rng_key)
  center = utils.mask_mean(
      mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6
  )
  rot = _random_rotation(rotation_key)
  translation = jax.random.normal(translation_key, shape=(3,))
  augmented = (
      jnp.einsum(
          '...i,ij->...j',
          positions - center,
          rot,
          precision=jax.lax.Precision.HIGHEST,
      )
      + translation
  )
  return augmented * mask[..., None]


def compute_differentiable_iptm(
    tmscore_adjusted_pae: jnp.ndarray,
    asym_id: jnp.ndarray,
    pair_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Differentiable iPTM from confidence head output.

  Uses a temperature-scaled softmax-weighted sum to approximate the max
  over per-alignment scores. This is always bounded by max(per_alignment),
  ensuring the output stays in [0, 1].

  Args:
    tmscore_adjusted_pae: [num_res, num_res] TM-score adjusted PAE values.
    asym_id: [num_res] chain IDs.
    pair_mask: [num_res, num_res] boolean mask for valid pairs.

  Returns:
    Scalar iPTM score in [0, 1].
  """
  # Interface mask: cross-chain pairs only
  interface_mask = pair_mask * (asym_id[:, None] != asym_id[None, :])

  # Normalized residue mask (per-row normalization)
  normed_mask = interface_mask / (
      1e-8 + jnp.sum(interface_mask, axis=-1, keepdims=True)
  )

  # Per-alignment score: for each residue i, average tmscore over its
  # interface contacts
  per_alignment = jnp.sum(tmscore_adjusted_pae * normed_mask, axis=-1)

  # Mask out residues with no interface contacts
  has_interface = jnp.sum(interface_mask, axis=-1) > 0.5

  # Softmax-weighted sum to approximate max (bounded by max(per_alignment))
  temperature = 10.0
  logits = jnp.where(has_interface, per_alignment * temperature, -1e9)
  weights = jax.nn.softmax(logits)
  score = jnp.sum(per_alignment * weights)
  return score


def compute_differentiable_ptm(
    tmscore_adjusted_pae: jnp.ndarray,
    pair_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Differentiable pTM from confidence head output.

  Uses a temperature-scaled softmax-weighted sum to approximate the max
  over per-alignment scores.

  Args:
    tmscore_adjusted_pae: [num_res, num_res] TM-score adjusted PAE values.
    pair_mask: [num_res, num_res] boolean mask for valid pairs.

  Returns:
    Scalar pTM score in [0, 1].
  """
  normed_mask = pair_mask / (
      1e-8 + jnp.sum(pair_mask, axis=-1, keepdims=True)
  )
  per_alignment = jnp.sum(tmscore_adjusted_pae * normed_mask, axis=-1)

  # Mask out residues with no valid pairs
  has_pairs = jnp.sum(pair_mask, axis=-1) > 0.5

  # Softmax-weighted sum to approximate max
  temperature = 10.0
  logits = jnp.where(has_pairs, per_alignment * temperature, -1e9)
  weights = jax.nn.softmax(logits)
  score = jnp.sum(per_alignment * weights)
  return score


def compute_differentiable_plddt(
    predicted_lddt: jnp.ndarray,
    atom_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Differentiable mean pLDDT from confidence head output.

  Args:
    predicted_lddt: [num_res, num_atoms] predicted LDDT values (0-100 scale).
    atom_mask: [num_res, num_atoms] mask for valid atoms.

  Returns:
    Scalar mean pLDDT score (0-100 scale).
  """
  masked_lddt = predicted_lddt * atom_mask
  score = jnp.sum(masked_lddt) / (jnp.sum(atom_mask) + 1e-8)
  return score


def _compute_d0(n_res: jnp.ndarray, min_value: float = 1.0) -> jnp.ndarray:
  """Compute d0 normalization factor from number of residues.

  d0 = max(min_value, 1.24 * (L - 15)^(1/3) - 1.8)

  Args:
    n_res: Number of residues (can be any shape).
    min_value: Minimum value for d0.

  Returns:
    d0 value(s) with same shape as n_res.
  """
  n_res_clipped = jnp.maximum(n_res.astype(jnp.float32), 19.0)
  d0 = 1.24 * jnp.power(n_res_clipped - 15.0, 1.0 / 3.0) - 1.8
  return jnp.maximum(d0, min_value)


def _soft_pae_mask(
    pae_values: jnp.ndarray, cutoff: float, temperature: float
) -> jnp.ndarray:
  """Create a differentiable soft mask for PAE values below cutoff.

  Uses sigmoid: sigmoid((cutoff - pae) / temperature) ≈ 1 when pae < cutoff.

  Args:
    pae_values: PAE values in Angstroms.
    cutoff: PAE cutoff threshold.
    temperature: Temperature for sigmoid (lower = sharper).

  Returns:
    Soft mask with values in [0, 1].
  """
  return jax.nn.sigmoid((cutoff - pae_values) / temperature)


def compute_differentiable_ipsae(
    full_pae: jnp.ndarray,
    asym_id: jnp.ndarray,
    seq_mask: jnp.ndarray,
    pae_cutoff: float = 10.0,
    soft_temperature: float = 1.0,
    d0_mode: str = 'd0res',
) -> jnp.ndarray:
  """Differentiable ipSAE from PAE values.

  ipSAE applies a PAE cutoff to focus on well-predicted interface contacts,
  then computes a TM-score-like rescaling. Unlike iPTM which uses all residue
  pairs, ipSAE only includes pairs with PAE below the cutoff.

  Based on: https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2

  Args:
    full_pae: [num_res, num_res] expected PAE values in Angstroms.
    asym_id: [num_res] chain IDs per residue.
    seq_mask: [num_res] mask for valid residues.
    pae_cutoff: PAE cutoff threshold (typically 10 Angstroms).
    soft_temperature: Temperature for soft PAE mask (lower = sharper).
    d0_mode: d0 normalization mode:
      - 'd0res': per-residue d0 from count of valid cross-chain partners
      - 'd0chn': d0 from total cross-chain residue count
      - 'd0dom': d0 from residues with good PAE values

  Returns:
    Scalar ipSAE score in [0, 1].
  """
  pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(jnp.float32)

  # Inter-chain mask
  inter_chain_mask = (
      (asym_id[:, None] != asym_id[None, :]).astype(jnp.float32) * pair_mask
  )

  # Soft PAE validity mask (differentiable)
  pae_valid_mask = _soft_pae_mask(full_pae, pae_cutoff, soft_temperature)

  # Combined mask: inter-chain AND valid PAE
  valid_pair_mask = inter_chain_mask * pae_valid_mask

  # Per-residue count of valid inter-chain partners
  row_valid_counts = jnp.sum(valid_pair_mask, axis=-1)  # [N]

  if d0_mode == 'd0res':
    # d0 based on per-residue valid pair count
    d0 = _compute_d0(jnp.maximum(row_valid_counts, 1.0))  # [N]
    # Broadcast d0 for the TM function: d0[i] applies to row i
    d0_expanded = d0[:, None]  # [N, 1]
  elif d0_mode == 'd0chn':
    # d0 based on total number of cross-chain residues
    # For each residue i in chain c, count all residues NOT in chain c
    n_cross_chain = jnp.sum(inter_chain_mask, axis=-1)  # [N]
    d0 = _compute_d0(n_cross_chain + seq_mask.astype(jnp.float32))  # [N]
    d0_expanded = d0[:, None]  # [N, 1]
  elif d0_mode == 'd0dom':
    # d0 based on residues with good PAE (valid pairs exist)
    # Count residues that have at least one valid cross-chain partner
    has_valid_row = (row_valid_counts > 0.5).astype(jnp.float32)  # [N]
    col_valid_counts = jnp.sum(valid_pair_mask, axis=-2)  # [N]
    has_valid_col = (col_valid_counts > 0.5).astype(jnp.float32)  # [N]
    n_dom = jnp.sum(has_valid_row) + jnp.sum(has_valid_col)
    d0 = _compute_d0(jnp.maximum(n_dom, 1.0))  # scalar
    d0_expanded = d0  # scalar broadcasts
  else:
    raise ValueError(f'Unknown d0_mode: {d0_mode}')

  # TM-score rescaling: 1 / (1 + (pae/d0)^2)
  tm_values = 1.0 / (1.0 + jnp.square(full_pae / d0_expanded))

  # Per-residue ipSAE: average TM values over valid cross-chain pairs
  per_res_ipsae = jnp.sum(
      tm_values * valid_pair_mask, axis=-1
  ) / (row_valid_counts + 1e-8)

  # Mask out residues with no valid pairs
  has_valid = row_valid_counts > 0.5

  # Soft-max over residues to approximate the max (differentiable)
  temperature = 10.0
  logits = jnp.where(has_valid, per_res_ipsae * temperature, -1e9)
  weights = jax.nn.softmax(logits)
  score = jnp.sum(per_res_ipsae * weights)

  return score


def parse_scoring_type(scoring_str: str) -> list[tuple[float, str]]:
  """Parse scoring type string into weighted components.

  Supports formats like:
    - "iptm"
    - "ptm"
    - "plddt"
    - "ipsae", "ipsae_d0res", "ipsae_d0chn", "ipsae_d0dom"
    - "0.8*iptm+0.2*ptm"
    - "0.5*ipsae+0.3*iptm+0.2*ptm"

  Args:
    scoring_str: String specification of the scoring function.

  Returns:
    List of (weight, metric_name) tuples.
  """
  scoring_str = scoring_str.strip()

  _VALID_METRICS = (
      'iptm', 'ptm', 'plddt',
      'ipsae', 'ipsae_d0res', 'ipsae_d0chn', 'ipsae_d0dom',
  )

  # Simple single metric
  if scoring_str in _VALID_METRICS:
    return [(1.0, scoring_str)]

  # Parse weighted combination like "0.8*iptm+0.2*ptm"
  components = []
  terms = re.findall(r'([0-9.]+)\*(\w+)', scoring_str)
  if not terms:
    raise ValueError(
        f'Could not parse scoring_type: {scoring_str}. '
        f'Expected format like "iptm", "0.8*iptm+0.2*ptm", etc. '
        f'Valid metrics: {_VALID_METRICS}'
    )
  for weight_str, metric in terms:
    weight = float(weight_str)
    if metric not in _VALID_METRICS:
      raise ValueError(
          f'Unknown metric: {metric}. Valid metrics: {_VALID_METRICS}'
      )
    components.append((weight, metric))
  return components


def compute_score(
    confidence_output: dict[str, jnp.ndarray],
    batch_dict: dict[str, Any],
    scoring_type: str,
    ipsae_pae_cutoff: float = 10.0,
    ipsae_soft_temperature: float = 1.0,
) -> jnp.ndarray:
  """Compute the scalar score from confidence head outputs.

  Args:
    confidence_output: Output dict from the confidence head (containing
      tmscore_adjusted_pae_interface, predicted_lddt, full_pae, etc.)
    batch_dict: The batch dictionary with token features.
    scoring_type: String specifying the scoring function.
    ipsae_pae_cutoff: PAE cutoff for ipSAE computation (Angstroms).
    ipsae_soft_temperature: Temperature for soft PAE mask in ipSAE.

  Returns:
    Scalar score value.
  """
  components = parse_scoring_type(scoring_type)

  from alphafold3.model import feat_batch  # pylint: disable=g-import-not-at-top
  batch = feat_batch.Batch.from_data_dict(batch_dict)
  seq_mask = batch.token_features.mask
  asym_id = batch.token_features.asym_id
  pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(jnp.bool_)

  # Map ipSAE variant names to d0_mode
  _IPSAE_D0_MODES = {
      'ipsae': 'd0res',
      'ipsae_d0res': 'd0res',
      'ipsae_d0chn': 'd0chn',
      'ipsae_d0dom': 'd0dom',
  }

  score = jnp.zeros(())
  for weight, metric in components:
    if metric == 'iptm':
      tmscore_pae = confidence_output['tmscore_adjusted_pae_interface']
      if tmscore_pae.ndim == 3:
        tmscore_pae = tmscore_pae[0]
      metric_score = compute_differentiable_iptm(
          tmscore_pae, asym_id, pair_mask
      )
    elif metric == 'ptm':
      tmscore_pae = confidence_output['tmscore_adjusted_pae_global']
      if tmscore_pae.ndim == 3:
        tmscore_pae = tmscore_pae[0]
      metric_score = compute_differentiable_ptm(tmscore_pae, pair_mask)
    elif metric == 'plddt':
      predicted_lddt = confidence_output['predicted_lddt']
      if predicted_lddt.ndim == 3:
        predicted_lddt = predicted_lddt[0]
      atom_mask = batch.predicted_structure_info.atom_mask
      metric_score = compute_differentiable_plddt(predicted_lddt, atom_mask)
      # Normalize to [0, 1] range for combination with other metrics
      metric_score = metric_score / 100.0
    elif metric in _IPSAE_D0_MODES:
      full_pae = confidence_output['full_pae']
      if full_pae.ndim == 3:
        full_pae = full_pae[0]
      metric_score = compute_differentiable_ipsae(
          full_pae=full_pae,
          asym_id=asym_id,
          seq_mask=seq_mask,
          pae_cutoff=ipsae_pae_cutoff,
          soft_temperature=ipsae_soft_temperature,
          d0_mode=_IPSAE_D0_MODES[metric],
      )
    else:
      raise ValueError(f'Unknown metric: {metric}')
    score = score + weight * metric_score

  return score


def clip_grads(
    grads: dict[str, jnp.ndarray], max_norm: float
) -> dict[str, jnp.ndarray]:
  """Clip gradients by global norm."""
  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree.leaves(grads))
  )
  scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
  return jax.tree.map(lambda g: g * scale, grads)


def init_adam_state(
    embeddings: dict[str, jnp.ndarray],
) -> dict[str, Any]:
  """Initialize Adam optimizer state.

  Keys starting with '_' are treated as constants and excluded from optimization.
  """
  # Filter out constant keys (starting with '_')
  opt_keys = {k: v for k, v in embeddings.items() if not k.startswith('_')}
  return {
      'm': jax.tree.map(jnp.zeros_like, opt_keys),
      'v': jax.tree.map(jnp.zeros_like, opt_keys),
      'step': 0,
  }


def adam_update(
    embeddings: dict[str, jnp.ndarray],
    grads: dict[str, jnp.ndarray],
    state: dict[str, Any],
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
  """Perform one Adam optimizer step.

  Keys starting with '_' are treated as constants and excluded from optimization.
  """
  step = state['step'] + 1

  # Filter out constant keys (starting with '_')
  opt_keys = [k for k in embeddings.keys() if not k.startswith('_')]
  opt_embeddings = {k: embeddings[k] for k in opt_keys}
  opt_grads = {k: grads[k] for k in opt_keys if k in grads}

  m = jax.tree.map(
      lambda m, g: beta1 * m + (1 - beta1) * g, state['m'], opt_grads
  )
  v = jax.tree.map(
      lambda v, g: beta2 * v + (1 - beta2) * g**2, state['v'], opt_grads
  )

  bc1 = 1 - beta1**step
  bc2 = 1 - beta2**step

  updated_opt = jax.tree.map(
      lambda e, m, v: e - learning_rate * (m / bc1) / (jnp.sqrt(v / bc2) + eps),
      opt_embeddings,
      m,
      v,
  )

  # Merge back with constant keys
  updated = {**embeddings, **updated_opt}

  new_state = {'m': m, 'v': v, 'step': step}
  return updated, new_state


def optimize_embeddings(
    model_config: Any,
    model_params: Any,
    batch: dict[str, Any],
    embeddings_init: dict[str, jnp.ndarray],
    opt_config: EmbeddingOptimizationConfig,
    rng_key: jnp.ndarray,
    device: Any,
    output_dir: str | None = None,
) -> dict[str, jnp.ndarray]:
  """Run per-step embedding optimization within the diffusion loop.

  The diffusion loop is unrolled manually. At each denoising step (controlled
  by optimization_interval), the confidence head is run on the current denoised
  positions, and gradients w.r.t. embeddings are used to update them. Updated
  embeddings are passed to the next denoising iteration.

  Args:
    model_config: Model configuration object.
    model_params: Haiku model parameters.
    batch: Featurised batch dictionary.
    embeddings_init: Initial embeddings from the evoformer trunk
      (keys: 'single', 'pair').
    opt_config: Optimization configuration.
    rng_key: JAX random key.
    device: JAX device for computation.
    output_dir: Optional directory to save optimization metrics CSV. If None,
      metrics are not saved to file.

  Returns:
    Optimized embeddings dictionary with 'single' and 'pair' keys.
  """
  from alphafold3.model import feat_batch  # pylint: disable=g-import-not-at-top
  from alphafold3.model import model as model_module  # pylint: disable=g-import-not-at-top
  from alphafold3.model.network import confidence_head as conf_head_module  # pylint: disable=g-import-not-at-top
  from alphafold3.model.network import diffusion_head as diff_head_module  # pylint: disable=g-import-not-at-top
  from alphafold3.model.network import template_modules  # pylint: disable=g-import-not-at-top

  # Patch dgram_from_positions with soft binning for differentiable gradients.
  # Store original function for restoration after optimization.
  _original_dgram_fn = template_modules.dgram_from_positions

  # Temperature for soft binning (lower = sharper, closer to hard binning)
  # Use the ipSAE soft temperature setting for consistency
  soft_dgram_temperature = opt_config.ipsae_soft_temperature

  def _soft_dgram_wrapper(positions, config):
    """Wrapper that uses soft binning for differentiable gradients."""
    return soft_dgram_from_positions_config(positions, config, soft_dgram_temperature)

  template_modules.dgram_from_positions = _soft_dgram_wrapper
  logging.info('Patched dgram_from_positions with soft binning (temp=%.2f) for optimization',
               soft_dgram_temperature)

  batch_obj = feat_batch.Batch.from_data_dict(batch)
  atom_mask = batch_obj.predicted_structure_info.atom_mask

  # Compute cross-chain mask if cross_chain_only is enabled
  cross_chain_mask = None
  if opt_config.cross_chain_only:
    asym_id = batch_obj.token_features.asym_id
    # Cross-chain pairs: asym_id[i] != asym_id[j]
    cross_chain_mask = (asym_id[:, None] != asym_id[None, :])
    logging.info(
        'Cross-chain only mode: %d/%d pairs are cross-chain (%.1f%%)',
        int(jnp.sum(cross_chain_mask)),
        cross_chain_mask.size,
        100.0 * float(jnp.mean(cross_chain_mask)),
    )

  # Determine number of diffusion steps
  diffusion_steps = opt_config.diffusion_steps
  if diffusion_steps <= 0:
    diffusion_steps = model_config.heads.diffusion.eval.steps

  # Noise schedule
  noise_levels = _noise_schedule(jnp.linspace(0, 1, diffusion_steps + 1))

  # -- Define haiku-transformed functions --
  # These use a custom hk.Module with name='diffuser' to match the parameter
  # hierarchy of the original Model class. This ensures that model_params
  # (obtained from the full model) can be used directly with apply().

  # 1. Denoise-only function (for steps without optimization)
  @hk.transform
  def denoise_fn(batch_dict, embeddings, positions_noisy, noise_level):
    """Run a single denoising step, return denoised positions."""

    class _DenoiseScope(hk.Module):
      """Wrapper matching Model's param hierarchy for denoiser access."""

      def __init__(self):
        super().__init__(name='diffuser')
        self.diffusion_module = diff_head_module.DiffusionHead(
            model_config.heads.diffusion, model_config.global_config
        )

      def __call__(self, batch_inner, emb, pos_noisy, nl):
        target_feat = model_module.create_target_feat_embedding(
            batch=batch_inner,
            config=model_config.evoformer,
            global_config=model_config.global_config,
        )
        full_embeddings = {
            'single': emb['single'],
            'pair': emb['pair'],
            'target_feat': target_feat,
        }
        return self.diffusion_module(
            positions_noisy=pos_noisy,
            noise_level=nl,
            batch=batch_inner,
            embeddings=full_embeddings,
            use_conditioning=True,
        )

    batch_inner = feat_batch.Batch.from_data_dict(batch_dict)
    return _DenoiseScope()(batch_inner, embeddings, positions_noisy, noise_level)

  # 2. Score-only function (confidence head on denoised positions)
  @hk.transform
  def score_fn(batch_dict, embeddings, denoised_positions):
    """Run confidence head on denoised positions, return scores."""

    class _ScoreScope(hk.Module):
      """Wrapper matching Model's param hierarchy for confidence head."""

      def __init__(self):
        super().__init__(name='diffuser')

      def __call__(self, batch_inner, emb, denoised):
        target_feat = model_module.create_target_feat_embedding(
            batch=batch_inner,
            config=model_config.evoformer,
            global_config=model_config.global_config,
        )
        full_embeddings = {
            'single': emb['single'],
            'pair': emb['pair'],
            'target_feat': target_feat,
        }
        confidence_output = conf_head_module.ConfidenceHead(
            model_config.heads.confidence, model_config.global_config
        )(
            dense_atom_positions=denoised,
            embeddings=full_embeddings,
            seq_mask=batch_inner.token_features.mask,
            token_atoms_to_pseudo_beta=(
                batch_inner.pseudo_beta_info.token_atoms_to_pseudo_beta
            ),
            asym_id=batch_inner.token_features.asym_id,
        )
        score = compute_score(
            confidence_output, batch_dict, opt_config.scoring_type,
            ipsae_pae_cutoff=opt_config.ipsae_pae_cutoff,
            ipsae_soft_temperature=opt_config.ipsae_soft_temperature,
        )

        # Compute individual iPTM, pTM, and ipSAE for logging
        seq_mask = batch_inner.token_features.mask
        asym_id = batch_inner.token_features.asym_id
        pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(jnp.bool_)
        tmscore_pae_interface = confidence_output[
            'tmscore_adjusted_pae_interface'
        ]
        if tmscore_pae_interface.ndim == 3:
          tmscore_pae_interface = tmscore_pae_interface[0]
        iptm = compute_differentiable_iptm(
            tmscore_pae_interface, asym_id, pair_mask
        )
        tmscore_pae_global = confidence_output['tmscore_adjusted_pae_global']
        if tmscore_pae_global.ndim == 3:
          tmscore_pae_global = tmscore_pae_global[0]
        ptm = compute_differentiable_ptm(tmscore_pae_global, pair_mask)

        # Compute ipSAE for logging
        full_pae = confidence_output['full_pae']
        if full_pae.ndim == 3:
          full_pae = full_pae[0]
        ipsae = compute_differentiable_ipsae(
            full_pae=full_pae,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pae_cutoff=opt_config.ipsae_pae_cutoff,
            soft_temperature=opt_config.ipsae_soft_temperature,
            d0_mode='d0res',
        )

        # Compute pLDDT for logging (separate from score to always have it)
        predicted_lddt = confidence_output['predicted_lddt']
        if predicted_lddt.ndim == 3:
          predicted_lddt = predicted_lddt[0]
        atom_mask = batch_inner.predicted_structure_info.atom_mask
        plddt = compute_differentiable_plddt(predicted_lddt, atom_mask)

        return score, iptm, ptm, ipsae, plddt

    batch_inner = feat_batch.Batch.from_data_dict(batch_dict)
    return _ScoreScope()(batch_inner, embeddings, denoised_positions)

  # 3. Pairformer projection function (optional, no backprop through it)
  # This expects params that have been extracted and re-keyed so that
  # trunk_pairformer params are directly under __layer_stack_no_per_layer/
  @hk.transform
  def pairformer_projection_fn(batch_dict, embeddings):
    """Apply Pairformer projection to embeddings without gradients.

    This runs the Pairformer (trunk) on the embeddings to project them back
    towards the learned embedding manifold. Uses all 48 Pairformer layers
    with pretrained weights (identical to the trunk).

    Args:
      batch_dict: Featurized batch dictionary.
      embeddings: Dict with 'single' and 'pair' keys.

    Returns:
      Projected embeddings dict with 'single' and 'pair' keys.
    """
    from alphafold3.model.network import modules  # pylint: disable=g-import-not-at-top

    batch_inner = feat_batch.Batch.from_data_dict(batch_dict)
    dtype = embeddings['pair'].dtype
    seq_mask = batch_inner.token_features.mask.astype(dtype)
    pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(dtype)

    pairformer_config = model_config.evoformer.pairformer

    pair_act = embeddings['pair']
    single_act = embeddings['single']

    # Define the pairformer iteration function
    def pairformer_fn(x):
      pairformer_iteration = modules.PairFormerIteration(
          pairformer_config,
          model_config.global_config,
          with_single=True,
          name='trunk_pairformer',
      )
      pair_in, single_in = x
      return pairformer_iteration(
          act=pair_in,
          single_act=single_in,
          pair_mask=pair_mask,
          seq_mask=seq_mask,
      )

    # layer_stack will create params under __layer_stack_no_per_layer/trunk_pairformer/
    # We've re-keyed the extracted params to match this structure
    pairformer_stack = hk.experimental.layer_stack(
        pairformer_config.num_layer
    )(pairformer_fn)

    pair_act, single_act = pairformer_stack((pair_act, single_act))

    return {'single': single_act, 'pair': pair_act}

  # -- Define loss and gradient functions --
  # Use jax.checkpoint on denoise_fn and score_fn separately so that
  # DiffusionHead and ConfidenceHead backward passes don't overlap in memory.
  # Peak memory = max(DiffusionHead_backward, ConfidenceHead_backward)
  # instead of sum(both).
  _denoise_apply = jax.checkpoint(
      denoise_fn.apply, prevent_cse=False
  )
  _score_apply = jax.checkpoint(
      score_fn.apply, prevent_cse=False
  )

  batch_size = opt_config.optimization_batch_size
  use_vmap_batch = batch_size > 1 and not opt_config.gradient_accumulation

  def step_loss_fn(opt_params, positions_noisy, noise_level, rng_key):
    """Loss for a single step: -score + anchor regularization.

    Args:
      opt_params: Structured optimization parameters (from init_update_params).
      positions_noisy: Current noisy atom positions. Shape [N, A, 3] when
        using single-sample path (batch_size=1 or gradient_accumulation=True),
        or [batch_size, N, A, 3] when using vmap batch path.
      noise_level: Current noise level scalar.
      rng_key: JAX random key.

    Returns:
      (loss, aux) where aux = (denoised, score, iptm, ptm, ipsae, plddt,
        anchor_s, anchor_z).
    """
    # Reconstruct full embeddings from structured params
    embeddings = apply_update(opt_params, embeddings_init, opt_config)

    if use_vmap_batch:
      # Split keys for batch denoising and scoring
      rng_keys = jax.random.split(rng_key, batch_size + 1)
      denoise_keys = rng_keys[:batch_size]
      score_key = rng_keys[batch_size]

      # vmap denoise over batch dim (positions vary, embeddings shared)
      batched_denoise = jax.vmap(
          lambda pos, key: _denoise_apply(
              model_params, key, batch, embeddings, pos, noise_level
          ),
          in_axes=(0, 0),
      )
      denoised_batch = batched_denoise(positions_noisy, denoise_keys)

      # vmap score over batch dim
      score_keys = jax.random.split(score_key, batch_size)
      batched_score = jax.vmap(
          lambda pos, key: _score_apply(
              model_params, key, batch, embeddings, pos
          ),
          in_axes=(0, 0),
      )
      scores_batch = batched_score(denoised_batch, score_keys)

      # Average scores across batch
      score = jnp.mean(scores_batch[0])
      iptm = jnp.mean(scores_batch[1])
      ptm = jnp.mean(scores_batch[2])
      ipsae = jnp.mean(scores_batch[3])
      plddt = jnp.mean(scores_batch[4])

      # Use first sample's denoised positions for Euler update
      denoised = denoised_batch[0]
    else:
      # Single-sample path (used for batch_size=1 or gradient_accumulation)
      rng_key_denoise, rng_key_score = jax.random.split(rng_key)

      # Stage 1: Denoise (checkpointed - activations recomputed during backward)
      denoised = _denoise_apply(
          model_params, rng_key_denoise, batch,
          embeddings, positions_noisy, noise_level
      )

      # Stage 2: Score (checkpointed separately - its backward doesn't overlap
      # with DiffusionHead backward in memory)
      (score, iptm, ptm, ipsae, plddt) = _score_apply(
          model_params, rng_key_score, batch, embeddings, denoised
      )

    # Anchor regularization using Frobenius norm on reconstructed embeddings.
    anchor_s = opt_config.anchor_lambda_single * jnp.sqrt(
        jnp.sum((embeddings['single'] - embeddings_init['single']) ** 2) + 1e-8
    )
    anchor_z = opt_config.anchor_lambda_pair * jnp.sqrt(
        jnp.sum((embeddings['pair'] - embeddings_init['pair']) ** 2) + 1e-8
    )
    loss = -score + anchor_s + anchor_z
    return loss, (denoised, score, iptm, ptm, ipsae, plddt, anchor_s, anchor_z)

  # JIT-compile the gradient function (grad w.r.t. first arg = opt_params)
  grad_fn = jax.jit(
      jax.grad(step_loss_fn, argnums=0, has_aux=True), device=device
  )

  # JIT-compile the denoise-only function
  denoise_jit = jax.jit(
      functools.partial(denoise_fn.apply, model_params), device=device
  )

  # JIT-compile the Pairformer projection function (if enabled)
  pairformer_projection_jit = None
  if opt_config.pairformer_projection_enabled:
    logging.info(
        'Pairformer projection enabled: interval=%d',
        opt_config.pairformer_projection_interval,
    )

    # Extract and re-key trunk_pairformer params.
    # Original path: diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/...
    # Target path:   __layer_stack_no_per_layer/trunk_pairformer/...
    # This is needed because layer_stack names are sequential (_1 is the 2nd),
    # but our projection only has one layer_stack so it gets no suffix.
    pairformer_params = {}
    source_prefix = None

    for key in model_params.keys():
      if '__layer_stack_no_per_layer' in key and 'trunk_pairformer' in key:
        # Find the layer_stack prefix for trunk_pairformer
        # e.g., 'diffuser/evoformer/__layer_stack_no_per_layer_1/'
        idx = key.index('trunk_pairformer')
        source_prefix = key[:idx]
        break

    if source_prefix is None:
      logging.warning(
          'Could not find trunk_pairformer layer_stack in model_params. '
          'Pairformer projection will be disabled.'
      )
      pairformer_projection_jit = None
    else:
      logging.info('Found trunk_pairformer source prefix: %s', source_prefix)

      # Determine the target prefix (what a fresh layer_stack would create)
      # Replace the numbered suffix with no suffix
      # e.g., '__layer_stack_no_per_layer_1' -> '__layer_stack_no_per_layer'
      target_prefix = '__layer_stack_no_per_layer/'

      for key, value in model_params.items():
        if key.startswith(source_prefix) and 'trunk_pairformer' in key:
          # Re-key: remove source_prefix, add target_prefix
          suffix = key[len(source_prefix):]  # e.g., 'trunk_pairformer/...'
          new_key = target_prefix + suffix
          pairformer_params[new_key] = value

      logging.info(
          'Extracted and re-keyed %d trunk_pairformer params',
          len(pairformer_params),
      )
      # Log sample keys for debugging
      sample_keys = list(pairformer_params.keys())[:3]
      logging.info('Sample re-keyed params: %s', sample_keys)

      def _projection_wrapper(rng_key, batch_dict, embeddings):
        return pairformer_projection_fn.apply(
            pairformer_params, rng_key, batch_dict, embeddings
        )

      pairformer_projection_jit = jax.jit(_projection_wrapper, device=device)

  # -- Initialize optimization state --
  # Ensure float32 for gradient precision (DiffusionHead uses bfloat16
  # internally, but embedding gradients should accumulate in float32).
  embeddings_init = {
      'single': jnp.asarray(embeddings_init['single'], dtype=jnp.float32),
      'pair': jnp.asarray(embeddings_init['pair'], dtype=jnp.float32),
  }

  # Initialize structured optimization parameters
  opt_params = init_update_params(embeddings_init, opt_config, cross_chain_mask)
  adam_state = init_adam_state(opt_params)

  logging.info(
      'Embedding optimization: update_mode=%s, num_params=%d, batch_size=%d,'
      ' grad_accum=%s, perturbation_cliff_pct=%.2f, score_threshold=%.2f,'
      ' scoring_type=%s, cross_chain_only=%s',
      opt_config.update_mode,
      sum(p.size for p in jax.tree.leaves(opt_params)),
      batch_size,
      opt_config.gradient_accumulation,
      opt_config.perturbation_cliff_pct,
      opt_config.score_threshold,
      opt_config.scoring_type,
      opt_config.cross_chain_only,
  )

  # Track whether the perturbation cliff has been reached.
  # Once reached, optimization stops and normal generation continues.
  cliff_reached = False

  # Track whether we've warned about structured mode + projection incompatibility
  structured_projection_warned = False

  # Track total number of embedding updates for Pairformer projection interval
  total_embedding_updates = 0

  # Collect optimization metrics for CSV export
  optimization_metrics = []

  # -- Outer diffusion loop --
  for outer_step in range(opt_config.outer_steps):
    logging.info('Embedding optimization: outer_step=%d', outer_step)

    # Initialize diffusion positions from noise
    rng_key, noise_key = jax.random.split(rng_key)
    positions = (
        jax.random.normal(noise_key, atom_mask.shape + (3,)) * noise_levels[0]
    )

    noise_level_prev = noise_levels[0]

    # Unrolled diffusion loop
    for step_idx in range(diffusion_steps):
      noise_level_next = noise_levels[step_idx + 1]

      # Random augmentation
      rng_key, aug_key = jax.random.split(rng_key)
      positions = _random_augmentation(aug_key, positions, atom_mask)

      # Stochastic noise injection (second-order correction)
      gamma = opt_config.gamma_0 * (noise_level_next > opt_config.gamma_min)
      t_hat = noise_level_prev * (1 + gamma)

      rng_key, noise_key = jax.random.split(rng_key)
      added_noise_scale = (
          opt_config.noise_scale * jnp.sqrt(t_hat**2 - noise_level_prev**2)
      )
      if batch_size > 1:
        # Generate batch_size noise realizations for robust gradient estimates
        noise_keys = jax.random.split(noise_key, batch_size)
        noise_batch = jnp.stack([
            added_noise_scale * jax.random.normal(k, positions.shape)
            for k in noise_keys
        ])
        positions_noisy = positions[None] + noise_batch  # [B, N, A, 3]
      else:
        noise = added_noise_scale * jax.random.normal(
            noise_key, positions.shape
        )
        positions_noisy = positions + noise

      # Should we run optimization at this step?
      # Skip optimization if cliff has already been reached.
      run_opt = (
          step_idx % opt_config.optimization_interval == 0
      ) and not cliff_reached

      # Reconstruct current embeddings for denoising
      embeddings = apply_update(opt_params, embeddings_init, opt_config)

      if run_opt:
        # Run optimization inner loop
        denoised = None
        use_grad_accum = batch_size > 1 and opt_config.gradient_accumulation
        for opt_step in range(opt_config.optimization_steps):
          rng_key, grad_key = jax.random.split(rng_key)

          if use_grad_accum:
            # Sequential gradient accumulation: call grad_fn once per sample,
            # accumulate gradients. Uses same memory as batch_size=1.
            acc_grads = None
            acc_score = jnp.zeros(())
            acc_iptm = jnp.zeros(())
            acc_ptm = jnp.zeros(())
            acc_ipsae = jnp.zeros(())
            acc_plddt = jnp.zeros(())
            denoised = None
            for b in range(batch_size):
              key_b = jax.random.fold_in(grad_key, b)
              g, (d, s, i, p, ip, pl, a_s, a_z) = grad_fn(
                  opt_params, positions_noisy[b], t_hat, key_b
              )
              if b == 0:
                denoised = d
                anchor_s, anchor_z = a_s, a_z
              if acc_grads is None:
                acc_grads = g
              else:
                acc_grads = jax.tree.map(jnp.add, acc_grads, g)
              acc_score = acc_score + s
              acc_iptm = acc_iptm + i
              acc_ptm = acc_ptm + p
              acc_ipsae = acc_ipsae + ip
              acc_plddt = acc_plddt + pl
            grads = jax.tree.map(lambda g: g / batch_size, acc_grads)
            score = acc_score / batch_size
            iptm = acc_iptm / batch_size
            ptm = acc_ptm / batch_size
            ipsae = acc_ipsae / batch_size
            plddt = acc_plddt / batch_size
          else:
            grads, (denoised, score, iptm, ptm, ipsae, plddt, anchor_s, anchor_z) = (
                grad_fn(opt_params, positions_noisy, t_hat, grad_key)
            )

          # Compute gradient norm before clipping
          grad_norm = jnp.sqrt(
              sum(jnp.sum(g**2) for g in jax.tree.leaves(grads))
          )

          # Gradient clipping
          grads = clip_grads(grads, opt_config.max_grad_norm)

          # Adam update on structured params
          opt_params, adam_state = adam_update(
              opt_params, grads, adam_state, opt_config.learning_rate
          )
          # Update embeddings after param update
          embeddings = apply_update(opt_params, embeddings_init, opt_config)

          # Increment total embedding updates counter
          total_embedding_updates += 1

          # Apply Pairformer projection if enabled and at the right interval.
          # The projection interval is based on total embedding updates across
          # all outer/diffusion/optimization steps, not per-inner-loop step.
          # The projection is applied with stop_gradient (no backprop) to
          # project embeddings back towards the learned embedding manifold.
          if (
              opt_config.pairformer_projection_enabled
              and pairformer_projection_jit is not None
              and total_embedding_updates % opt_config.pairformer_projection_interval == 0
          ):
            rng_key, proj_key = jax.random.split(rng_key)
            # Apply projection with stop_gradient to prevent backprop
            projected_embeddings = pairformer_projection_jit(
                proj_key, batch, jax.lax.stop_gradient(embeddings)
            )
            # Stop gradient on the projected embeddings as well
            projected_embeddings = jax.tree.map(
                jax.lax.stop_gradient, projected_embeddings
            )
            logging.info(
                'Applied Pairformer projection at total_embedding_updates=%d',
                total_embedding_updates,
            )
            # For 'full' update mode, directly update opt_params so the
            # projection persists to subsequent optimization steps.
            # Note: anchor regularization still uses the original embeddings_init
            # from the closure, which is correct behavior (anchor pulls back to
            # the original embeddings while projection keeps them on manifold).
            if opt_config.update_mode == 'full':
              opt_params['pair'] = projected_embeddings['pair']
              if opt_config.optimize_single:
                opt_params['single'] = projected_embeddings['single']
              embeddings = projected_embeddings
            else:
              # For structured modes (low_rank, element_scale, etc.), we cannot
              # easily update opt_params to match the projected embeddings
              # (would require solving an optimization problem to find the best
              # structured approximation). Instead, we update `embeddings` which
              # affects only the current denoising step - the next optimization
              # step will use the unprojected (structured) embeddings.
              # For best results with projection, use update_mode='full'.
              if not structured_projection_warned:
                logging.warning(
                    'Pairformer projection with structured update_mode=%s: '
                    'projection affects denoising but does not persist to '
                    'next optimization step. For persistent projection, use '
                    'update_mode=full or set pairformer_projection_interval '
                    'to match optimization_steps.',
                    opt_config.update_mode,
                )
                structured_projection_warned = True
              embeddings = projected_embeddings

          # Compute perturbation as % of original norm
          pair_diff = embeddings['pair'] - embeddings_init['pair']
          delta_z = jnp.sqrt(jnp.sum(pair_diff ** 2))
          z_init_norm = jnp.sqrt(jnp.sum(embeddings_init['pair'] ** 2) + 1e-8)
          z_curr_norm = jnp.sqrt(jnp.sum(embeddings['pair'] ** 2) + 1e-8)
          delta_z_pct = delta_z / z_init_norm * 100.0

          # Debug: check if embeddings are actually different
          same_object = embeddings['pair'] is embeddings_init['pair']
          max_abs_diff = jnp.max(jnp.abs(pair_diff))
          if same_object:
            logging.warning('BUG: embeddings["pair"] is same object as embeddings_init["pair"]!')

          delta_s = jnp.sqrt(
              jnp.sum((embeddings['single'] - embeddings_init['single']) ** 2)
          )
          s_init_norm = jnp.sqrt(
              jnp.sum(embeddings_init['single'] ** 2) + 1e-8
          )
          delta_s_pct = delta_s / s_init_norm * 100.0

          # Compute beta norm for element_scale modes
          beta_norm = jnp.zeros(())
          if 'pair_beta' in opt_params:
            beta_norm = jnp.sqrt(
                jnp.sum(opt_params['pair_beta'] ** 2)
            )

          logging.info(
              'diff_step=%d/%d opt_step=%d | iptm=%.4f ptm=%.4f '
              'ipsae=%.4f plddt=%.2f score=%.4f anchor_s=%.6f anchor_z=%.6f '
              'loss=%.4f grad_norm=%.6f delta_z=%.6f%% delta_s=%.4f%% '
              'z_init_norm=%.2f z_curr_norm=%.2f norm_diff=%.6f '
              'max_elem_diff=%.8f beta_norm=%.8f',
              step_idx, diffusion_steps, opt_step,
              float(iptm), float(ptm), float(ipsae), float(plddt), float(score),
              float(anchor_s), float(anchor_z),
              float(-score + anchor_s + anchor_z),
              float(grad_norm),
              float(delta_z_pct), float(delta_s_pct),
              float(z_init_norm), float(z_curr_norm),
              float(z_curr_norm - z_init_norm), float(max_abs_diff),
              float(beta_norm),
          )

          # Collect metrics for CSV export
          optimization_metrics.append({
              'outer_step': outer_step,
              'diff_step': step_idx,
              'opt_step': opt_step,
              'iptm': float(iptm),
              'ptm': float(ptm),
              'ipsae': float(ipsae),
              'plddt': float(plddt),
              'score': float(score),
              'loss': float(-score + anchor_s + anchor_z),
              'grad_norm': float(grad_norm),
              'delta_z_pct': float(delta_z_pct),
              'delta_s_pct': float(delta_s_pct),
              'beta_norm': float(beta_norm),
              'anchor_s': float(anchor_s),
              'anchor_z': float(anchor_z),
          })

          # Check if perturbation cliff has been reached.
          # Use delta_z_pct for full/low_rank modes, beta_norm for element_scale.
          if opt_config.perturbation_cliff_pct > 0.0:
            if opt_config.update_mode in (
                'element_scale', 'element_scale_channel',
                'element_scale_residue', 'element_scale_residue_channel',
            ):
              # For element_scale modes, use beta norm as % (beta starts near 0)
              # The effective perturbation is |beta| since z = (1+beta)*z_init.
              # Convert to percentage: for small beta, |delta_z|/|z_init| ≈ |beta|.
              effective_pct = float(beta_norm) * 100.0
            else:
              # For full/low_rank modes, use direct embedding perturbation %
              effective_pct = float(delta_z_pct)

            if effective_pct >= opt_config.perturbation_cliff_pct:
              cliff_reached = True
              logging.info(
                  'Perturbation cliff reached: %.4f%% >= %.4f%%. '
                  'Stopping optimization, continuing normal generation.',
                  effective_pct, opt_config.perturbation_cliff_pct,
              )
              # Break out of optimization inner loop
              break

          # Check if score threshold has been reached.
          if opt_config.score_threshold < 1.0:
            if float(score) >= opt_config.score_threshold:
              cliff_reached = True
              logging.info(
                  'Score threshold reached: %.4f >= %.4f (scoring_type=%s). '
                  'Stopping optimization, continuing normal generation.',
                  float(score), opt_config.score_threshold,
                  opt_config.scoring_type,
              )
              # Break out of optimization inner loop
              break
      else:
        # Run denoiser without optimization (single sample only)
        rng_key, denoise_key = jax.random.split(rng_key)
        positions_noisy_single = (
            positions_noisy[0] if batch_size > 1 else positions_noisy
        )
        denoised = denoise_jit(
            denoise_key, batch, embeddings, positions_noisy_single, t_hat
        )

      # Euler update: x_{t+1} = x_t + step_scale * dt * grad_estimate
      # Use first sample's noisy positions when batch_size > 1
      positions_noisy_euler = (
          positions_noisy[0] if batch_size > 1 else positions_noisy
      )
      grad_estimate = (positions_noisy_euler - denoised) / t_hat
      d_t = noise_level_next - t_hat
      positions = (
          positions_noisy_euler + opt_config.step_scale * d_t * grad_estimate
      )

      noise_level_prev = noise_level_next

  # Write optimization metrics to CSV if output_dir is provided
  if output_dir is not None and optimization_metrics:
    csv_path = os.path.join(output_dir, 'embedding_optimization_metrics.csv')
    fieldnames = [
        'outer_step', 'diff_step', 'opt_step', 'iptm', 'ptm', 'ipsae', 'plddt',
        'score', 'loss', 'grad_norm', 'delta_z_pct', 'delta_s_pct',
        'beta_norm', 'anchor_s', 'anchor_z',
    ]
    try:
      with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(optimization_metrics)
      logging.info('Saved optimization metrics to %s', csv_path)
    except IOError as e:
      logging.warning('Failed to save optimization metrics CSV: %s', e)

  # Restore original hard binning for final evaluation
  template_modules.dgram_from_positions = _original_dgram_fn
  logging.info('Restored original dgram_from_positions (hard binning) for final evaluation')

  # Return the final reconstructed embeddings
  return apply_update(opt_params, embeddings_init, opt_config)
