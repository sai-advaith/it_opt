# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Coordinate guidance for AlphaFold 3 diffusion sampling.

Implements classifier-guidance-style steering of the diffusion process by
computing gradients of confidence scores (iPTM/PTM/pLDDT) w.r.t. denoised
coordinates and applying a drift term to nudge coordinates toward higher
confidence.

Unlike embedding optimization which modifies the trunk embeddings, coordinate
guidance directly modifies the denoised coordinates at each diffusion step:

  for diff_step in range(diffusion_steps):
    positions_noisy = augment_and_noise(positions, ...)
    denoised = denoise(embeddings, positions_noisy, noise_level)
    if guidance_enabled:
      grad_coords = grad(score(denoised), denoised)
      denoised = denoised + guidance_scale * grad_coords
    positions = euler_update(positions_noisy, denoised, ...)

This is a simpler baseline compared to embedding optimization as it doesn't
require optimizing parameters across diffusion steps.
"""

import csv
import dataclasses
import functools
import os
from typing import Any

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp

from alphafold3.model import embedding_optimization


# From diffusion_head.py
_SIGMA_DATA = 16.0


def soft_dgram_from_positions(positions, min_bin=3.25, max_bin=50.75, num_bins=39,
                               temperature=1.0):
  """Compute soft distogram from positions with differentiable binning.

  Uses sigmoid functions instead of hard comparisons to allow gradient flow.

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
class CoordinateGuidanceConfig:
  """Configuration for coordinate guidance during diffusion."""

  enabled: bool = False
  # Guidance scale: how much to push coordinates toward higher confidence.
  # Higher values = stronger guidance. Typical values: 0.1 - 10.0
  guidance_scale: float = 1.0
  # Number of diffusion denoising steps. Set to 0 to use model default.
  diffusion_steps: int = 0
  # Apply guidance every N diffusion steps (1 = every step).
  guidance_interval: int = 1
  # Start applying guidance after this fraction of diffusion is complete.
  # E.g., 0.5 means only apply guidance in the second half of diffusion.
  guidance_start_fraction: float = 0.0
  # Stop applying guidance after this fraction of diffusion.
  # E.g., 0.8 means stop guidance in the last 20% of diffusion.
  guidance_stop_fraction: float = 1.0
  # Scoring function for guidance. Options: "iptm", "ptm", "plddt",
  # "contact" (direct geometric), "pae_proxy" (direct geometric),
  # or weighted combinations like "0.8*iptm+0.2*ptm".
  # The confidence head scoring now uses soft binning for differentiability.
  scoring_type: str = 'iptm'
  # Maximum gradient norm for gradient clipping.
  max_grad_norm: float = 10.0
  # ipSAE parameters (if using ipSAE in scoring)
  ipsae_pae_cutoff: float = 10.0
  ipsae_soft_temperature: float = 1.0
  # Diffusion sampling parameters (overrides model defaults)
  gamma_0: float = 0.8
  gamma_min: float = 1.0
  noise_scale: float = 1.003
  step_scale: float = 1.5
  # Number of gradient steps per diffusion step (for iterative refinement).
  # Default 1 means single gradient step; higher values iteratively refine.
  gradient_steps: int = 1
  # Learning rate for iterative refinement (only used if gradient_steps > 1).
  # With gradient_steps=1, the full gradient is applied scaled by guidance_scale.
  learning_rate: float = 0.1
  # Whether to normalize the gradient before applying (unit direction).
  normalize_gradient: bool = False
  # Scale guidance based on noise level (higher noise = less guidance).
  # If True, guidance_scale is multiplied by (sigma_data / noise_level).
  noise_adaptive_scale: bool = False
  # Contact score parameters (for scoring_type="contact")
  contact_threshold: float = 8.0  # Angstroms
  contact_temperature: float = 1.0  # Softness of contact function


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
  from alphafold3.model.components import utils
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


def compute_interface_contact_score(
    positions: jnp.ndarray,
    asym_id: jnp.ndarray,
    seq_mask: jnp.ndarray,
    contact_threshold: float = 8.0,
    temperature: float = 1.0,
) -> jnp.ndarray:
  """Compute differentiable interface contact score from coordinates.

  This bypasses the confidence head and computes a direct score based on
  inter-chain contacts. Higher score = more/closer interface contacts.

  For single-chain structures (no interface), falls back to a compactness
  score that rewards closer non-local contacts.

  Args:
    positions: (num_res, 3) CA/pseudo-beta positions.
    asym_id: (num_res,) chain IDs.
    seq_mask: (num_res,) mask for valid residues.
    contact_threshold: Distance threshold for contacts (Angstroms).
    temperature: Temperature for soft contact function.

  Returns:
    Scalar contact score in [0, 1].
  """
  # Compute pairwise distances
  diff = positions[:, None, :] - positions[None, :, :]  # [N, N, 3]
  dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)  # [N, N]

  # Soft contact probability: sigmoid((threshold - dist) / temp)
  contact_prob = jax.nn.sigmoid((contact_threshold - dist) / temperature)

  pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(jnp.float32)

  # Check if multi-chain (interface exists)
  interface_mask = (asym_id[:, None] != asym_id[None, :]).astype(jnp.float32)
  interface_mask = interface_mask * pair_mask
  num_interface_pairs = jnp.sum(interface_mask)

  # For single-chain: use non-local contacts (sequence separation > 12)
  n_res = positions.shape[0]
  seq_idx = jnp.arange(n_res)
  seq_sep = jnp.abs(seq_idx[:, None] - seq_idx[None, :])
  nonlocal_mask = (seq_sep > 12).astype(jnp.float32) * pair_mask

  # Use interface mask if available, otherwise use non-local mask
  use_interface = num_interface_pairs > 0.5
  scoring_mask = jnp.where(use_interface, interface_mask, nonlocal_mask)
  num_pairs = jnp.sum(scoring_mask) + 1e-8

  score = jnp.sum(contact_prob * scoring_mask) / num_pairs

  return score


def compute_pae_proxy_score(
    positions: jnp.ndarray,
    asym_id: jnp.ndarray,
    seq_mask: jnp.ndarray,
    d0_factor: float = 1.24,
) -> jnp.ndarray:
  """Compute differentiable PAE-like proxy score from coordinates.

  Uses the TM-score formula applied to pairwise distances as a proxy for PAE.
  This assumes well-predicted structures have consistent inter-residue distances.

  Args:
    positions: (num_res, 3) CA/pseudo-beta positions.
    asym_id: (num_res,) chain IDs.
    seq_mask: (num_res,) mask for valid residues.
    d0_factor: Scaling factor for d0 computation.

  Returns:
    Scalar score in [0, 1], higher = better.
  """
  # Compute pairwise distances
  diff = positions[:, None, :] - positions[None, :, :]
  dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)

  # Compute d0 based on sequence length (TM-score normalization)
  n_res = jnp.sum(seq_mask)
  n_res_clipped = jnp.maximum(n_res, 19.0)
  d0 = d0_factor * jnp.power(n_res_clipped - 15.0, 1.0 / 3.0) - 1.8
  d0 = jnp.maximum(d0, 0.5)

  # TM-score-like term: 1 / (1 + (d/d0)^2)
  # For well-folded structures, this should be high for nearby residues
  tm_term = 1.0 / (1.0 + jnp.square(dist / d0))

  # Interface mask for cross-chain scoring
  interface_mask = (asym_id[:, None] != asym_id[None, :]).astype(jnp.float32)
  pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(jnp.float32)
  interface_mask = interface_mask * pair_mask

  # Score = mean TM term over interface pairs
  num_interface_pairs = jnp.sum(interface_mask) + 1e-8
  score = jnp.sum(tm_term * interface_mask) / num_interface_pairs

  return score


def clip_grads_by_norm(grads: jnp.ndarray, max_norm: float) -> jnp.ndarray:
  """Clip gradient array by its norm."""
  grad_norm = jnp.sqrt(jnp.sum(grads**2) + 1e-8)
  scale = jnp.minimum(1.0, max_norm / grad_norm)
  return grads * scale


def run_guided_diffusion(
    model_config: Any,
    model_params: Any,
    batch: dict[str, Any],
    embeddings: dict[str, jnp.ndarray],
    guidance_config: CoordinateGuidanceConfig,
    rng_key: jnp.ndarray,
    device: Any,
    output_dir: str | None = None,
) -> dict[str, jnp.ndarray]:
  """Run diffusion sampling with coordinate guidance.

  This function unrolls the diffusion loop and at each denoising step:
  1. Denoise the noisy positions to get clean coordinates
  2. Compute the confidence score on denoised coordinates
  3. Compute gradient of score w.r.t. denoised coordinates
  4. Add the gradient (scaled) to the denoised coordinates as guidance

  Args:
    model_config: Model configuration object.
    model_params: Haiku model parameters.
    batch: Featurised batch dictionary.
    embeddings: Trunk embeddings from the evoformer ('single', 'pair' keys).
    guidance_config: Coordinate guidance configuration.
    rng_key: JAX random key.
    device: JAX device for computation.
    output_dir: Optional directory to save guidance metrics CSV.

  Returns:
    Dictionary with 'atom_positions' (num_samples, num_tokens, max_atoms, 3)
    and 'mask' arrays, matching the format of diffusion_head.sample().
  """
  from alphafold3.model import feat_batch
  from alphafold3.model import model as model_module
  from alphafold3.model.network import confidence_head as conf_head_module
  from alphafold3.model.network import diffusion_head as diff_head_module

  batch_obj = feat_batch.Batch.from_data_dict(batch)
  atom_mask = batch_obj.predicted_structure_info.atom_mask

  # Determine number of diffusion steps
  diffusion_steps = guidance_config.diffusion_steps
  if diffusion_steps <= 0:
    diffusion_steps = model_config.heads.diffusion.eval.steps

  num_samples = model_config.heads.diffusion.eval.num_samples

  # Noise schedule
  noise_levels = _noise_schedule(jnp.linspace(0, 1, diffusion_steps + 1))

  # Compute guidance window (which steps to apply guidance)
  start_step = int(guidance_config.guidance_start_fraction * diffusion_steps)
  stop_step = int(guidance_config.guidance_stop_fraction * diffusion_steps)

  # -- Define haiku-transformed functions --

  # 1. Denoise function
  @hk.transform
  def denoise_fn(batch_dict, emb, positions_noisy, noise_level):
    """Run a single denoising step, return denoised positions."""

    class _DenoiseScope(hk.Module):
      def __init__(self):
        super().__init__(name='diffuser')
        self.diffusion_module = diff_head_module.DiffusionHead(
            model_config.heads.diffusion, model_config.global_config
        )

      def __call__(self, batch_inner, emb_inner, pos_noisy, nl):
        target_feat = model_module.create_target_feat_embedding(
            batch=batch_inner,
            config=model_config.evoformer,
            global_config=model_config.global_config,
        )
        full_embeddings = {
            'single': emb_inner['single'],
            'pair': emb_inner['pair'],
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
    return _DenoiseScope()(batch_inner, emb, positions_noisy, noise_level)

  # 2. Score function (confidence head on denoised positions)
  # We need to use soft binning for differentiability
  from alphafold3.model.network import template_modules

  # Store original function
  _original_dgram_fn = template_modules.dgram_from_positions

  # Temperature for soft binning (lower = sharper, closer to hard binning)
  soft_dgram_temperature = guidance_config.contact_temperature

  def _soft_dgram_wrapper(positions, config):
    """Wrapper that uses soft binning for differentiable gradients."""
    return soft_dgram_from_positions_config(positions, config, soft_dgram_temperature)

  @hk.transform
  def score_fn(batch_dict, emb, denoised_positions):
    """Run confidence head on denoised positions, return score and metrics.

    NOTE: This function expects template_modules.dgram_from_positions to be
    patched with the soft version for gradient computation.
    """

    class _ScoreScope(hk.Module):
      def __init__(self):
        super().__init__(name='diffuser')

      def __call__(self, batch_inner, emb_inner, denoised):
        target_feat = model_module.create_target_feat_embedding(
            batch=batch_inner,
            config=model_config.evoformer,
            global_config=model_config.global_config,
        )
        full_embeddings = {
            'single': emb_inner['single'],
            'pair': emb_inner['pair'],
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
        score = embedding_optimization.compute_score(
            confidence_output, batch_dict, guidance_config.scoring_type,
            ipsae_pae_cutoff=guidance_config.ipsae_pae_cutoff,
            ipsae_soft_temperature=guidance_config.ipsae_soft_temperature,
        )

        # Compute individual metrics for logging
        seq_mask = batch_inner.token_features.mask
        asym_id = batch_inner.token_features.asym_id
        pair_mask = (seq_mask[:, None] * seq_mask[None, :]).astype(jnp.bool_)

        tmscore_pae_interface = confidence_output['tmscore_adjusted_pae_interface']
        if tmscore_pae_interface.ndim == 3:
          tmscore_pae_interface = tmscore_pae_interface[0]
        iptm = embedding_optimization.compute_differentiable_iptm(
            tmscore_pae_interface, asym_id, pair_mask
        )

        tmscore_pae_global = confidence_output['tmscore_adjusted_pae_global']
        if tmscore_pae_global.ndim == 3:
          tmscore_pae_global = tmscore_pae_global[0]
        ptm = embedding_optimization.compute_differentiable_ptm(
            tmscore_pae_global, pair_mask
        )

        full_pae = confidence_output['full_pae']
        if full_pae.ndim == 3:
          full_pae = full_pae[0]
        ipsae = embedding_optimization.compute_differentiable_ipsae(
            full_pae=full_pae,
            asym_id=asym_id,
            seq_mask=seq_mask,
            pae_cutoff=guidance_config.ipsae_pae_cutoff,
            soft_temperature=guidance_config.ipsae_soft_temperature,
            d0_mode='d0res',
        )

        return score, iptm, ptm, ipsae

    batch_inner = feat_batch.Batch.from_data_dict(batch_dict)
    return _ScoreScope()(batch_inner, emb, denoised_positions)

  # Patch the dgram function for soft binning (needed for gradient computation)
  template_modules.dgram_from_positions = _soft_dgram_wrapper
  logging.info('Patched dgram_from_positions with soft binning (temp=%.2f)',
               soft_dgram_temperature)

  # -- Define gradient function --
  # Check if using direct geometric scoring (differentiable) vs confidence head
  use_direct_scoring = guidance_config.scoring_type in ('contact', 'pae_proxy')

  if use_direct_scoring:
    # Direct geometric score - fully differentiable
    seq_mask = batch_obj.token_features.mask
    asym_id = batch_obj.token_features.asym_id

    # Check if multi-chain
    unique_chains = jnp.unique(asym_id * seq_mask)
    num_chains = jnp.sum(unique_chains > 0) + (unique_chains[0] == 0)
    logging.info('Direct scoring: num_chains=%d, num_tokens=%d',
                 int(num_chains), int(jnp.sum(seq_mask)))

    def direct_score_fn(denoised_positions):
      """Compute direct geometric score from positions.

      Uses mean atom position per token as representative coordinate.
      This avoids atom_layout.convert which may have tracing issues.
      """
      # denoised_positions: (num_tokens, max_atoms_per_token, 3)
      # Use first atom (usually N or representative) for each token
      # Or use mean of all atoms per token
      token_pos = denoised_positions[:, 0, :]  # (num_tokens, 3)

      if guidance_config.scoring_type == 'contact':
        score = compute_interface_contact_score(
            token_pos, asym_id, seq_mask,
            contact_threshold=guidance_config.contact_threshold,
            temperature=guidance_config.contact_temperature,
        )
      else:  # pae_proxy
        score = compute_pae_proxy_score(
            token_pos, asym_id, seq_mask,
        )
      return score

    grad_direct_fn = jax.jit(jax.grad(direct_score_fn), device=device)
    direct_score_jit = jax.jit(direct_score_fn, device=device)

  # Compute gradient of score w.r.t. denoised positions (confidence head path)
  def score_for_grad(denoised_positions, rng_key):
    """Score function for gradient computation (returns scalar)."""
    score, _, _, _ = score_fn.apply(
        model_params, rng_key, batch, embeddings, denoised_positions
    )
    return score

  # JIT-compile functions
  denoise_jit = jax.jit(
      functools.partial(denoise_fn.apply, model_params), device=device
  )
  score_jit = jax.jit(
      functools.partial(score_fn.apply, model_params), device=device
  )
  grad_score_fn = jax.jit(
      jax.grad(score_for_grad, argnums=0), device=device
  )

  # Ensure embeddings are float32
  embeddings = {
      'single': jnp.asarray(embeddings['single'], dtype=jnp.float32),
      'pair': jnp.asarray(embeddings['pair'], dtype=jnp.float32),
  }

  logging.info(
      'Coordinate guidance: scale=%.4f, scoring=%s, steps=%d, '
      'window=[%.1f%%, %.1f%%], interval=%d, grad_steps=%d',
      guidance_config.guidance_scale,
      guidance_config.scoring_type,
      diffusion_steps,
      guidance_config.guidance_start_fraction * 100,
      guidance_config.guidance_stop_fraction * 100,
      guidance_config.guidance_interval,
      guidance_config.gradient_steps,
  )

  # Collect guidance metrics for CSV export
  guidance_metrics = []

  # -- Run guided diffusion for each sample --
  # Wrap in try/finally to ensure original dgram function is restored
  all_positions = []

  try:
    for sample_idx in range(num_samples):
      logging.info('Coordinate guidance: sample %d/%d', sample_idx + 1, num_samples)

      # Initialize positions from noise
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

        # Stochastic noise injection
        gamma = guidance_config.gamma_0 * (noise_level_next > guidance_config.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        rng_key, noise_key = jax.random.split(rng_key)
        added_noise_scale = (
            guidance_config.noise_scale * jnp.sqrt(t_hat**2 - noise_level_prev**2)
        )
        noise = added_noise_scale * jax.random.normal(noise_key, positions.shape)
        positions_noisy = positions + noise

        # Denoise
        rng_key, denoise_key = jax.random.split(rng_key)
        denoised = denoise_jit(denoise_key, batch, embeddings, positions_noisy, t_hat)

        # Check if we should apply guidance at this step
        apply_guidance = (
            step_idx >= start_step
            and step_idx < stop_step
            and step_idx % guidance_config.guidance_interval == 0
        )

        if apply_guidance:
          # Compute guidance scale (optionally adaptive to noise level)
          if guidance_config.noise_adaptive_scale:
            scale = guidance_config.guidance_scale * (_SIGMA_DATA / t_hat)
          else:
            scale = guidance_config.guidance_scale

          # Iterative gradient refinement
          for grad_step in range(guidance_config.gradient_steps):
            rng_key, score_key, grad_key = jax.random.split(rng_key, 3)

            if use_direct_scoring:
              # Direct geometric scoring - fully differentiable
              score = direct_score_jit(denoised)
              grad_coords = grad_direct_fn(denoised)

              # Debug: numerical gradient check on first step
              if step_idx == start_step and grad_step == 0:
                eps = 0.1
                denoised_pert = denoised.at[0, 0, 0].add(eps)
                score_pert = direct_score_jit(denoised_pert)
                numerical_grad = (score_pert - score) / eps
                logging.info(
                    'DEBUG numerical grad check: score=%.6f, score_pert=%.6f, '
                    'numerical_grad=%.6f, analytic_grad=%.6f',
                    float(score), float(score_pert), float(numerical_grad),
                    float(grad_coords[0, 0, 0]),
                )

              # Debug: check gradient statistics on first atom only
              first_atom_grad = grad_coords[:, 0, :]
              first_atom_grad_norm = jnp.sqrt(jnp.sum(first_atom_grad**2) + 1e-10)
              logging.info(
                  'DEBUG: score=%.6f, first_atom_grad_norm=%.6f, '
                  'first_atom_grad_max=%.6f, denoised_range=[%.2f, %.2f]',
                  float(score), float(first_atom_grad_norm),
                  float(jnp.max(jnp.abs(first_atom_grad))),
                  float(jnp.min(denoised)), float(jnp.max(denoised)),
              )

              # For logging, set confidence head metrics to the direct score
              iptm = score
              ptm = score
              ipsae = score
            else:
              # Confidence head scoring (using soft binning for differentiability)
              score, iptm, ptm, ipsae = score_jit(score_key, batch, embeddings, denoised)
              grad_coords = grad_score_fn(denoised, grad_key)

              # Debug: numerical gradient check on first step
              if step_idx == start_step and grad_step == 0:
                eps = 0.1
                denoised_pert = denoised.at[0, 0, 0].add(eps)
                score_pert, _, _, _ = score_jit(score_key, batch, embeddings, denoised_pert)
                numerical_grad = (score_pert - score) / eps
                logging.info(
                    'DEBUG numerical grad check (confidence head): score=%.6f, '
                    'score_pert=%.6f, numerical_grad=%.6f, analytic_grad=%.6f',
                    float(score), float(score_pert), float(numerical_grad),
                    float(grad_coords[0, 0, 0]),
                )

            # Compute gradient statistics
            grad_norm = jnp.sqrt(jnp.sum(grad_coords**2))
            grad_max = jnp.max(jnp.abs(grad_coords))

            # Clip gradient
            if guidance_config.max_grad_norm > 0:
              grad_coords = clip_grads_by_norm(grad_coords, guidance_config.max_grad_norm)

            # Optionally normalize gradient
            if guidance_config.normalize_gradient:
              grad_coords = grad_coords / (jnp.sqrt(jnp.sum(grad_coords**2)) + 1e-8)

            # Apply gradient update
            if guidance_config.gradient_steps > 1:
              # Iterative refinement with learning rate
              denoised = denoised + guidance_config.learning_rate * grad_coords
            else:
              # Single step with full guidance scale
              denoised = denoised + scale * grad_coords

            # Apply atom mask
            denoised = denoised * atom_mask[..., None]

            logging.info(
                'sample=%d diff_step=%d/%d grad_step=%d | score=%.4f '
                'grad_norm=%.6f grad_max=%.6f scale=%.4f scoring=%s',
                sample_idx, step_idx, diffusion_steps, grad_step,
                float(score), float(grad_norm), float(grad_max), float(scale),
                guidance_config.scoring_type,
            )

            # Collect metrics
            guidance_metrics.append({
                'sample': sample_idx,
                'diff_step': step_idx,
                'grad_step': grad_step,
                'iptm': float(iptm),
                'ptm': float(ptm),
                'ipsae': float(ipsae),
                'score': float(score),
                'grad_norm': float(grad_norm),
                'grad_max': float(grad_max),
                'scale': float(scale),
                'noise_level': float(t_hat),
            })

        # Euler update
        grad_estimate = (positions_noisy - denoised) / t_hat
        d_t = noise_level_next - t_hat
        positions = positions_noisy + guidance_config.step_scale * d_t * grad_estimate

        noise_level_prev = noise_level_next

      all_positions.append(positions)

    # Stack all samples
    positions_out = jnp.stack(all_positions, axis=0)
    mask_out = jnp.tile(atom_mask[None], (num_samples, 1, 1))

  finally:
    # Restore original hard binning for final evaluation
    template_modules.dgram_from_positions = _original_dgram_fn
    logging.info('Restored original dgram_from_positions (hard binning) for final evaluation')

  # Write metrics to CSV if output_dir is provided
  if output_dir is not None and guidance_metrics:
    csv_path = os.path.join(output_dir, 'coordinate_guidance_metrics.csv')
    fieldnames = [
        'sample', 'diff_step', 'grad_step', 'iptm', 'ptm', 'ipsae',
        'score', 'grad_norm', 'grad_max', 'scale', 'noise_level',
    ]
    try:
      os.makedirs(output_dir, exist_ok=True)
      with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(guidance_metrics)
      logging.info('Saved guidance metrics to %s', csv_path)
    except IOError as e:
      logging.warning('Failed to save guidance metrics CSV: %s', e)

  return {'atom_positions': positions_out, 'mask': mask_out}
