import torch
import os
from src.utils.non_diffusion_model_manager import ProtenixModelManager, MSA
import argparse
from src.utils.io import load_config, namespace_to_dict, query_msa_server
import wandb
from src.losses import *
from tqdm import tqdm

import pandas as pd
import shutil
import numpy as np
import time
from copy import deepcopy

import gemmi
from src.protenix.metrics.rmsd import self_aligned_rmsd
from src.utils.pdb_parsing import find_bonded_pairs

parser = argparse.ArgumentParser()

parser.add_argument("--configuration_file", type=str, required=False, help="the path to the configuration file", default="pipeline_configurations/nmr_baseline.yaml")
parser.add_argument('--device', type=str, required=False, default="cuda:0")

class ITOptimizationManager:
    def __init__(self, config, device):
        # general
        self.device = device
        self.config = config
        self.name = self.config.general.name
        self.query_msa_server()
        self.model_manager = self._get_model_manager()
        self._setup_wandb()
        self.loss_function = self._get_main_loss_function()

        # Adversarial loss function
        self.adversarial_loss_function = self._get_adversarial_loss_function()

        self.experiment_save_dir = os.path.join(self.config.general.output_folder, self.config.general.name)
        os.makedirs(self.experiment_save_dir, exist_ok=True)

        torch.manual_seed(self.config.general.seed)

    def _setup_wandb(self):
        if self.config.wandb.mode != "disabled" and self.config.wandb.login_key is not None:
            wandb.login(key=self.config.wandb.login_key)
            wandb.init(project=self.config.wandb.project ,name=self.name, mode=self.config.wandb.mode, config=namespace_to_dict(self.config))

    def _get_model_manager(self):
        return ProtenixModelManager(
            self.config.protein.sequence,
            self.config.protein.pdb_id,
            self.config.protein.reference_pdb,
            self.config.model_manager.N_cycle,
            self.config.model_manager.chunk_size,
            self.config.model_manager.diffusion_N,
            self.config.model_manager.gamma0,
            self.config.model_manager.gamma_min,
            self.config.model_manager.noise_scale_lambda,
            self.config.model_manager.step_scale_eta,
            "fp32",
            False,
            self.config.model_manager.msa_save_dir,
            self.config.model_manager.msa_embedding_cache_dir,
            self.config.model_manager.model_checkpoint_path,
            self.config.model_manager.dump_dir,
            self.config.model_manager.use_msa,
            self.config.general.batch_size,
            device=self.device
            )

    def _get_adversarial_loss_function(self):
        init_msa_feature = self.get_msa_features()
        anchor_lambda_s, anchor_lambda_z = float(self.config.loss_function.anchor_loss_function.lambda_s), float(self.config.loss_function.anchor_loss_function.lambda_z)
        init_s, init_z = init_msa_feature["s_trunk"], init_msa_feature["z_trunk"]

        anchor_loss_function = AnchorLossFunction(init_s, init_z, self.config.protein.pdb_id, anchor_lambda_s, anchor_lambda_z, self.device)
        return anchor_loss_function

    def _get_main_loss_function(self):
        # The main objective is selected by `loss_function.loss_function_type`.
        # Currently only the NMR restraint loss is supported.
        loss_type = getattr(self.config.loss_function, "loss_function_type", "nmr")
        if loss_type == "nmr":
            return self._get_nmr_loss_function()
        raise ValueError(
            f"Unsupported loss_function_type: {loss_type!r} (only 'nmr' is supported)"
        )

    def _get_nmr_loss_function(self):
        # NOE (+ optional methyl/amide RDC and relaxation) restraint loss.
        # Reads the parsed restraint CSV and the reference structure; the atom
        # array comes from the model manager (already built during setup).
        # Config keys match the pipeline_configurations/nmr_baseline.yaml block
        # (`reference_nmr` / `pdb_file`); `restraint_file` / `reference_pdb` and
        # protein.reference_pdb are accepted as fallbacks. Empty strings ("")
        # for the optional RDC/relaxation files are normalized to None so the
        # loss skips them.
        nmr_config = self.config.loss_function.nmr_loss_function
        restraint_file = getattr(nmr_config, "reference_nmr", None) or getattr(nmr_config, "restraint_file", None)
        pdb_file = (getattr(nmr_config, "pdb_file", None)
                    or getattr(nmr_config, "reference_pdb", None)
                    or self.config.protein.reference_pdb)
        return NMRLossFunction(
            restraint_file=restraint_file,
            pdb_file=pdb_file,
            atom_array=self.model_manager.atom_array,
            device=self.device,
            iid_loss=getattr(nmr_config, "iid_loss", False),
            noe_scale=float(getattr(nmr_config, "noe_scale", 1.0)),
            methyl_relax_file=getattr(nmr_config, "methyl_relax_file", None) or None,
            methyl_relax_scale=float(getattr(nmr_config, "methyl_relax_scale", 0.0)),
            methyl_rdc_file=getattr(nmr_config, "methyl_rdc_file", None) or None,
            methyl_rdc_scale=float(getattr(nmr_config, "methyl_rdc_scale", 0.0)),
            amide_rdc_file=getattr(nmr_config, "amide_rdc_file", None) or None,
            amide_rdc_scale=float(getattr(nmr_config, "amide_rdc_scale", 0.0)),
            amide_relax_file=getattr(nmr_config, "amide_relax_file", None) or None,
            amide_relax_scale=float(getattr(nmr_config, "amide_relax_scale", 0.0)),
        )

    def query_msa_server(self):
        query_msa_server(self.config.model_manager.msa_save_dir, self.config.protein.pdb_id, self.config.protein.sequence)

    def save_state(self, structures, name, folder_path, j=None):
        os.makedirs(folder_path, exist_ok=True)
        for i in range(structures.shape[0]):
            if j is not None:
                self.model_manager.save_structure_cif(structures[i].cpu(), f"{folder_path}/{name}_{i}_{j}.cif")
            else:
                self.model_manager.save_structure_cif(structures[i].cpu(), f"{folder_path}/{name}_{i}.cif")


    def get_initial_latents(self):
        return self.model_manager.get_x_noisy(self.model_manager.get_x_start(self.config.general.batch_size))

    def get_msa_features(self):
        # Take in latents noise and then output the msa single and pairwise features
        msa_features = self.model_manager.msa
        msa_feature_dict = {
            "s_inputs": msa_features.s_inputs,
            "s_trunk": msa_features.s,
            "z_trunk": msa_features.z,
        }
        return msa_feature_dict

    def get_confidence_scores(self, structures):
        # plddt vector
        with torch.no_grad():
            return self.model_manager.get_confidence_scores(structures)

    def run_full_diffusion_process(self, latents):
        structures = latents.clone()
        steps_generator = tqdm(range(self.config.model_manager.diffusion_N), "running diffusion process")

        # Diffusion loop
        with torch.no_grad():
            for i in steps_generator:
                x_0_hat = self.model_manager.get_x_0_hat_from_x_noisy(structures, start_index=i, inplace_safe=False)
                structures = self.model_manager.get_x_t_from_x_0_hat(structures, x_0_hat, i, i+1, guidance_direction=None)
                if i < self.config.model_manager.diffusion_N - 1:
                    structures = self.model_manager.get_x_noisy(structures, i + 1)
        return structures

    def calculate_within_chain_clash(self, coordinates, threshold=1.2):
        distances = torch.cdist(coordinates, coordinates, compute_mode="donot_use_mm_for_euclid_dist")
        distances = distances.triu(diagonal=1)
        distances = distances[distances > 0]
        loss = (torch.relu(threshold - distances)/0.25).exp()
        num_violations = (distances < threshold).sum()
        return loss.mean(), num_violations

    def debug_unguided_diffusion_process(self, optimized_msa_dict):
        with torch.no_grad():
            new_structures = self.get_initial_latents()
            optimized_msa = MSA(optimized_msa_dict["s_inputs"], optimized_msa_dict["s_trunk"], optimized_msa_dict["z_trunk"])
            sub_folder_name = "debug_unguided_diffusion_single"

            diffusion_steps = tqdm(range(self.config.model_manager.diffusion_N), "running unguided diffusion process")
            loss_values = []
            for diffusion_step in diffusion_steps:
                x_0_hat = self.model_manager.get_x_0_hat_from_x_noisy(new_structures, start_index=diffusion_step, inplace_safe=False, msa=optimized_msa)
                loss_values.append(self.loss_function(x_0_hat, None)[0])
                diffusion_steps.set_description(f"unguided diffusion {diffusion_step}, loss={loss_values[-1]}")
                new_structures = self.model_manager.get_x_t_from_x_0_hat(new_structures, x_0_hat, diffusion_step, diffusion_step+1, guidance_direction=None)
                if diffusion_step < self.config.model_manager.diffusion_N - 1:
                    new_structures = self.model_manager.get_x_noisy(new_structures, diffusion_step + 1)
            
            self.violation_loss_function(new_structures, None)
            violation_loss = self.violation_loss_function._last_loss
            # bond_length_loss, _ = self.bond_length_loss_function(new_structures, None)
            self.bond_length_loss_function(new_structures, None)
            bond_length_loss = self.bond_length_loss_function._last_bond_length_loss
            collision_loss = self.bond_length_loss_function._last_collision_loss
            print(f"Violation loss: {violation_loss}, Bond length loss: {bond_length_loss}, Collision loss: {collision_loss}")
            for frame in range(new_structures.shape[0]):
                _, num_violations = self.calculate_within_chain_clash(new_structures[frame].cpu())
                print(f"Structure {frame}, number of violations: {num_violations}")
                self.model_manager.save_structure_pdb(new_structures[frame].cpu(), f"{os.path.join(self.experiment_save_dir, sub_folder_name)}/{self.model_manager.pdb_id}_{frame}.pdb")
            return new_structures

    def run_unguided_batch_diffusion_process(self, msa_dict, log_step_loss=True, log_structure=True):
        with torch.no_grad():
            new_structures = self.get_initial_latents()
            init_msa_features = MSA(msa_dict["s_inputs"], msa_dict["s_trunk"], msa_dict["z_trunk"])
            init_msa_features.requires_grad(False)

            # Get batch size from config
            optimized_s_inputs = msa_dict["s_inputs"].clone().detach()  # [N_tokens, c_s_inputs]
            optimized_s = msa_dict["s_trunk"].clone().detach()  # [N_tokens, c_s]
            optimized_z = msa_dict["z_trunk"].clone().detach()  # [N_tokens, N_tokens, c_z]

            loss_values = []
            steps_generator = tqdm(range(self.config.model_manager.diffusion_N), "running unguided batch diffusion process")
            for i in steps_generator:
                x_0_hat = self.model_manager.get_x_0_hat_from_x_noisy_batched(
                    new_structures,
                    start_index=i,
                    inplace_safe=False,
                    s_inputs=optimized_s_inputs,
                    s_trunk=optimized_s,
                    z_trunk=optimized_z)
                if log_step_loss:
                    loss_values.append(self.loss_function(x_0_hat, None)[0])
                    steps_generator.set_description(f"unguided batch diffusion {i}, loss={loss_values[-1]}")
                else:
                    steps_generator.set_description(f"unguided batch diffusion {i}")

                new_structures = self.model_manager.get_x_t_from_x_0_hat(
                    new_structures,
                    x_0_hat,
                    i,
                    i+1,
                    guidance_direction=None)
                if i < self.config.model_manager.diffusion_N - 1:
                    new_structures = self.model_manager.get_x_noisy(new_structures, i + 1)
            sub_folder_name = "unguided_batch_diffusion"
            final_loss_value = (self.loss_function(new_structures, None)[0]).item()

            self.violation_loss_function(new_structures, None)
            violation_loss = self.violation_loss_function._last_loss
            self.bond_length_loss_function(new_structures, None)
            bond_length_loss, collision_loss = self.bond_length_loss_function._last_bond_length_loss, self.bond_length_loss_function._last_collision_loss
            print(f"Violation loss: {violation_loss}, Bond length loss: {bond_length_loss}, Collision loss: {collision_loss}")

            # Save everything
            total_violations = 0
            for frame in range(new_structures.shape[0]):
                _, num_violations = self.calculate_within_chain_clash(new_structures[frame].cpu())
                total_violations += num_violations
                print(f"Structure {frame}, number of violations: {num_violations}")
                if log_structure:
                    self.model_manager.save_structure_pdb(new_structures[frame].cpu(), f"{os.path.join(self.experiment_save_dir, sub_folder_name)}/{self.model_manager.pdb_id}_{frame}.pdb")

            return new_structures, final_loss_value, violation_loss, bond_length_loss, collision_loss, float(total_violations)/new_structures.shape[0]

    def run_diffusion_process_it_optimized(self, optimized_dict):
        with torch.no_grad():
            structures = self.get_initial_latents()
            msa_inputs = MSA(optimized_dict["s_inputs"], optimized_dict["s_trunk"], optimized_dict["z_trunk"])
            sub_folder_name = "diffusion_process"

            self.model_manager.msa = msa_inputs
            steps_generator = tqdm(range(self.config.model_manager.diffusion_N), "running unguided diffusion process")
            for i in steps_generator:
                x_0_hat = self.model_manager.get_x_0_hat_from_x_noisy(structures, start_index=i, inplace_safe=False, msa=msa_inputs)
                structures = self.model_manager.get_x_t_from_x_0_hat(structures, x_0_hat, i, i+1, guidance_direction=None)
                if i < self.config.model_manager.diffusion_N - 1:
                    structures = self.model_manager.get_x_noisy(structures, i + 1)

            # Save ensemble
            self.save_state(structures.clone().cpu().detach(), self.config.protein.pdb_id, os.path.join(self.experiment_save_dir, sub_folder_name))
            loss_value = (self.loss_function(structures, None)[0]).item()
            print(f"Loss value: {loss_value}")
            return structures


    def it_validation(self, optimized_dict):
        _, final_loss_value, violation_loss, bond_length_loss, collision_loss, avg_violations = self.run_unguided_batch_diffusion_process(optimized_dict, log_step_loss=False, log_structure=False)
        val_log_dict = {
            "val_final_loss_value": final_loss_value,
            "val_violation_loss": violation_loss,
            "val_bond_length_loss": bond_length_loss,
            "val_collision_loss": collision_loss,
            "val_avg_violations": avg_violations,
        }
        return val_log_dict

    def get_loss_values(self, x0_hat, diffusion_loop_iteration, diffusion_step, optimized_s_inputs, optimized_s, optimized_z):
        """
        Collect losses, gradients, and other values from the loss functions
        ALL LOSSES ARE SCALED!
        """
        current_msa = (optimized_s, optimized_z)
        main_loss, _ = self.loss_function(x0_hat, time=diffusion_step, msa=None)

        # Adversarial loss
        adversarial_loss, _ = self.adversarial_loss_function(x0_hat, None, msa=current_msa)
        adversarial_loss_s, adversarial_loss_z = adversarial_loss
        total_loss = main_loss + adversarial_loss_s + adversarial_loss_z

        # Wandb log
        main_wandb_log = self.loss_function.wandb_log(x0_hat)
        adversarial_wandb_log = self.adversarial_loss_function.wandb_log(x0_hat)

        # Combine all wandb logs
        wandb_log_dict = {}
        prefix = f"diffusion_loop_{diffusion_loop_iteration}"

        # Add all keys from main_wandb_log
        for key, value in main_wandb_log.items():
            wandb_log_dict[f"{prefix}/{key}"] = value

        # Add all keys from adversarial_wandb_log
        for key, value in adversarial_wandb_log.items():
            wandb_log_dict[f"{prefix}/{key}"] = value

        wandb_log_dict[f"{prefix}/total_loss"] = total_loss.item()
        wandb_log_dict[f"{prefix}/iteration"] = diffusion_step

        return total_loss, wandb_log_dict, main_loss

    def run_it_optimization(self):
        msa_features_t = self.get_msa_features()
        init_msa_features = MSA(msa_features_t["s_inputs"], msa_features_t["s_trunk"], msa_features_t["z_trunk"])
        init_msa_features.requires_grad(False)

        # Get batch size from config
        batch_size = self.config.general.batch_size
        
        init_s_inputs = msa_features_t["s_inputs"].clone().detach()  # [N_tokens, c_s_inputs]
        init_s_trunk = msa_features_t["s_trunk"].clone().detach()  # [N_tokens, c_s]
        init_z_pair = msa_features_t["z_trunk"].clone().detach()  # [N_tokens, N_tokens, c_z]

        # Expand to batch dimension by unsqueezing and expanding
        checkpoint = None
        if checkpoint is None:
            optimized_s_inputs = init_s_inputs.unsqueeze(0).expand(batch_size, -1, -1).clone().detach()  # [batch_size, N_tokens, c_s_inputs]
            optimized_s = init_s_trunk.unsqueeze(0).expand(batch_size, -1, -1).clone().detach()  # [batch_size, N_tokens, c_s]
            optimized_z = init_z_pair.unsqueeze(0).expand(batch_size, -1, -1, -1).clone().detach()  # [batch_size, N_tokens, N_tokens, c_z]
        else:
            print("Loading checkpoint")
            optimized_s_inputs = checkpoint["s_inputs"].clone().detach()
            optimized_s = checkpoint["s_trunk"].clone().detach()
            optimized_z = checkpoint["z_trunk"].clone().detach()

        optimized_s.requires_grad = True
        optimized_z.requires_grad = True

        # Number of diffusion loops to optimize s,z over
        num_full_diffusion_optimization_steps = self.config.it_optimization_parameters.outer_diffusion_steps
        max_grad_norm = self.config.it_optimization_parameters.max_grad_norm
        learning_rate = float(self.config.it_optimization_parameters.learning_rate)
        timestep = time.strftime('%Y:%m:%d_%H:%M:%S')

        for diffusion_loop_iteration in range(num_full_diffusion_optimization_steps):
            structures = self.get_initial_latents()
            structures.requires_grad = False

            # MSA dictionary - save batched embeddings
            msa_dict_t = {"s_inputs": optimized_s_inputs.clone().detach(), "s_trunk": optimized_s.clone().detach(), "z_trunk": optimized_z.clone().detach()}

            # Logs
            log_directory = os.path.join(self.experiment_save_dir, f"diffusion_optimized_msa_{diffusion_loop_iteration}_{timestep}")
            os.makedirs(log_directory, exist_ok=True)
            diffusion_steps = tqdm(range(self.config.model_manager.diffusion_N), "running diffusion process")
            torch.save(msa_dict_t, os.path.join(log_directory, f"msa_dict_init_{diffusion_loop_iteration}.pt"))

            # Run diffusion loop
            for diffusion_step in diffusion_steps:
                # Keep it small, or even lower for now
                optimizer = torch.optim.Adam([optimized_s, optimized_z], lr=learning_rate)
                noisy_structures = structures.clone().detach()

                optimizer.zero_grad()

                # denoised x0_hat
                # Use denoise_net_batched to handle batched embeddings
                # Each structure in the batch uses its corresponding embedding pair
                it_optim_x0_hat = self.model_manager.denoise_net_batched(
                    x_noisy=noisy_structures,
                    t_hat_noise_level=self.model_manager.get_t_hat(diffusion_step).reshape([1]).repeat(noisy_structures.shape[0]),
                    input_feature_dict=self.model_manager.input_feature_dict,
                    s_inputs=optimized_s_inputs,
                    s_trunk=optimized_s,
                    z_trunk=optimized_z,
                    chunk_size=self.model_manager.chunk_size,
                    inplace_safe=False,
                )
                # Loss
                total_loss, wandb_log_dict, main_loss = self.get_loss_values(it_optim_x0_hat, diffusion_loop_iteration, diffusion_step, optimized_s_inputs, optimized_s, optimized_z)
                total_loss.backward()

                # Clip grad norms
                torch.nn.utils.clip_grad_norm_([optimized_s], max_grad_norm)
                torch.nn.utils.clip_grad_norm_([optimized_z], max_grad_norm)

                # Tqdm step loss
                print(f"diffusion loop {diffusion_loop_iteration}, diffusion step {diffusion_step}, loss: {main_loss.item()}")

                # Log to wandb
                optimizer.step()
                if self.config.wandb.mode != "disabled" and self.config.wandb.login_key is not None:
                    wandb.log(wandb_log_dict)
                noisy_structures = noisy_structures.clone().detach()

                # Diffusion steps
                with torch.no_grad():
                    msa_dict_t = {"s_inputs": optimized_s_inputs.clone().detach(), "s_trunk": optimized_s.clone().detach(), "z_trunk": optimized_z.clone().detach()}

                    # Use batched version to ensure each structure uses its corresponding embedding
                    x_0_hat = self.model_manager.get_x_0_hat_from_x_noisy_batched(
                        structures, 
                        start_index=diffusion_step, 
                        inplace_safe=False,
                        s_inputs=optimized_s_inputs.clone().detach(),
                        s_trunk=optimized_s.clone().detach(),
                        z_trunk=optimized_z.clone().detach()
                    )

                    # Save the msa dictionary
                    # Diffusion_step between 160 and 170
                    if (diffusion_step + 1) == 160:
                        torch.save(msa_dict_t, os.path.join(log_directory, f"msa_dict_{diffusion_loop_iteration}_diffusion_step_{diffusion_step+1}.pt"))

                    # Leave this untouched.
                    structures = self.model_manager.get_x_t_from_x_0_hat(structures, x_0_hat, diffusion_step, diffusion_step+1)
                    # Take next step
                    if diffusion_step < self.config.model_manager.diffusion_N - 1:
                        structures = self.model_manager.get_x_noisy(structures, diffusion_step + 1)

        # Optimized MSA features
        optimized_dict = {"s_inputs": optimized_s_inputs, "s_trunk": optimized_s, "z_trunk": optimized_z}
        return optimized_dict

def main():
    # Load all representations into a tensor here
    args = parser.parse_args()
    configuration_file = args.configuration_file
    print(f"Configuration file: {configuration_file}")
    device = args.device
    config = load_config(configuration_file)
    pipeline = ITOptimizationManager(config, device)

    optimized_dict = pipeline.run_it_optimization()

if __name__ == "__main__":
    main()

