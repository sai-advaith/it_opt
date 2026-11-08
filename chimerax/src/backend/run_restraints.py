"""IT-opt of Protenix trunk embeddings against user geometric restraints.

    python run_restraints.py spec.json [--protenix /path/to/it_opt/protenix]

The spec (written by the ChimeraX `itopt` bundle) contains the chains to
predict, the restraints (see geom_restraints.py) and run settings.  Progress
is written atomically to <output_dir>/status.json; structures are written as
mmCIF with the user's chain ids and residue numbering:

    baseline_s{k}.cif   unguided Protenix samples (before optimisation)
    iter{i:03d}_s{k}.cif  samples at the end of outer loop i (guided trajectory)
    final_s{k}.cif      unguided samples from the optimised embeddings
    metrics.json        restraint values for baseline and final samples

The optimisation loop mirrors it_optimization_manager.run_it_optimization in
sai-advaith/it_opt: per-sample (s, z) embeddings, one Adam step per diffusion
step on the denoised x0_hat, anchor (L2) regulariser to the trunk embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROTENIX = os.path.expanduser("~/Documents/code/ext/it_opt/protenix")

STOP = False


def _on_term(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGINT, _on_term)


class Status:
    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "status.json")
        self.state = {"stage": "starting", "pid": os.getpid(), "t_start": time.time(), "files": []}
        self.write()

    def update(self, **kw):
        self.state.update(kw)
        self.state["t_now"] = time.time()
        self.write()

    def add_files(self, files):
        self.state["files"] = self.state.get("files", []) + list(files)
        self.write()

    def write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(tmp, self.path)


def seq_key(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--protenix", default=os.environ.get("ITOPT_PROTENIX", DEFAULT_PROTENIX))
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    out_dir = os.path.abspath(spec["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    status = Status(out_dir)
    try:
        run(spec, out_dir, os.path.abspath(args.protenix), status)
    except Exception as e:  # report to ChimeraX
        status.update(stage="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        raise


def run(spec, out_dir, protenix_dir, status):
    global STOP
    S = {
        "samples": 4,
        "outer": 5,
        "diffusion_steps": 100,
        "n_cycle": 4,
        "lr": 6e-2,
        "max_grad_norm": 0.6,
        "use_msa": True,
        "lambda_s": 1e-4,
        "lambda_z": 1e-3,
        "device": "mps",
        "seed": 0,
        "guide_from_step": 0,
        "restraint_scale": 1.0,
        "baseline": True,
        "cache_dir": os.path.expanduser("~/.cache/itopt-chimerax"),
    }
    S.update(spec.get("settings", {}))

    # ---- import the it_opt Protenix fork (expects cwd = protenix dir) ------
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.chdir(protenix_dir)
    sys.path[:0] = [protenix_dir, os.path.join(protenix_dir, "src"), HERE]
    status.update(stage="loading model")
    import torch
    import numpy as np
    import src.utils.non_diffusion_model_manager as ndm
    from src.utils.non_diffusion_model_manager import ProtenixModelManager, MSA
    from src.utils.io import query_msa_server
    from geom_restraints import AtomIndexer, GeometricRestraintLoss
    from write_cif import write_cif

    torch.manual_seed(int(S["seed"]))
    device = S["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    chains = spec["chains"]
    labels = [c["id"] for c in chains]
    cache = S["cache_dir"]
    msa_dir = os.path.join(cache, "msa")
    emb_dir = os.path.join(cache, "trunk")
    os.makedirs(msa_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    # ---- MSAs (ColabFold server, cached per sequence) ----------------------
    msa_keys = []
    for c in chains:
        k = "seq_" + seq_key(c["sequence"])
        msa_keys.append(k)
        if S["use_msa"] and not os.path.exists(os.path.join(msa_dir, k)):
            status.update(stage=f"querying MSA server for chain {c['id']}")
            query_msa_server(msa_dir, k, c["sequence"])

    class Manager(ProtenixModelManager):
        def _generate_msa_configuration(self):
            seqs = []
            for c, k in zip(chains, msa_keys):
                pc = {"sequence": c["sequence"], "count": 1}
                if S["use_msa"]:
                    pc["msa"] = {"precomputed_msa_dir": os.path.join(msa_dir, k, "msa"), "pairing_db": "uniref100"}
                seqs.append({"proteinChain": pc})
            return [{"sequences": seqs, "modelSeeds": [], "assembly_id": "1", "name": self.pdb_id}]

    # reference_pdb is only used by the NMR pipeline
    _orig_load = ndm.load_pdb_atom_locations
    ndm.load_pdb_atom_locations = lambda p, *a, **k: torch.zeros(1, 1, 3) if p is None else _orig_load(p, *a, **k)

    emb_key = seq_key(*[c["sequence"] for c in chains], S["use_msa"], S["n_cycle"])
    emb_file = os.path.join(emb_dir, emb_key + ".pt")
    have_emb = os.path.exists(emb_file)
    status.update(stage="building features" + ("" if have_emb else " + trunk (pairformer)"))
    t0 = time.time()
    B = int(S["samples"])
    N = int(S["diffusion_steps"])
    mm = Manager(
        "+".join(c["sequence"] for c in chains), "itopt_" + emb_key,
        reference_pdb=None, N_cycle=int(S["n_cycle"]), chunk_size=256, diffusion_N=N,
        msa_save_dir=msa_dir, msa_embedding_cache_dir=emb_dir,
        model_checkpoint_path=os.path.join(protenix_dir, "src/af3-dev/release_model/model_v0.2.0.pt"),
        dump_dir=os.path.join(cache, "protenix_dump"), use_msa=bool(S["use_msa"]),
        batch_size=B, device=device, initialize_msa=not have_emb,
    )
    dev = mm.runner.device
    if have_emb:
        d = torch.load(emb_file, map_location=dev)
        mm.msa = MSA(d["s_inputs"], d["s"], d["z"])
    else:
        torch.save({"s_inputs": mm.msa.s_inputs.cpu(), "s": mm.msa.s.cpu(), "z": mm.msa.z.cpu()}, emb_file)
    status.update(stage="model ready", t_setup=time.time() - t0, n_atoms=int(mm.atom_array.shape[0]),
                  n_tokens=int(mm.msa.s.shape[-2]), device=str(dev))

    # ---- restraints ---------------------------------------------------------
    indexer = AtomIndexer(mm.atom_array, labels)
    rloss = GeometricRestraintLoss(spec["restraints"], indexer, dev, scale=float(S["restraint_scale"]))
    s_inputs0 = mm.msa.s_inputs.detach()
    s0, z0 = mm.msa.s.detach(), mm.msa.z.detach()
    lam_s, lam_z = float(S["lambda_s"]), float(S["lambda_z"])

    def anchor(s, z):
        # per-sample L2 distance to the trunk embedding (it_opt AnchorLossFunction)
        ds = (s - s0.unsqueeze(0)).flatten(1).norm(dim=1).mean()
        dz = (z - z0.unsqueeze(0)).flatten(1).norm(dim=1).mean()
        return lam_s * ds, lam_z * dz

    res_numbers = {c["id"]: c.get("res_numbers") for c in chains}

    def save(structs, prefix):
        files = []
        for k in range(structs.shape[0]):
            fn = os.path.join(out_dir, f"{prefix}_s{k}.cif")
            write_cif(mm.atom_array, structs[k].detach().cpu().numpy(), labels, res_numbers, fn, title=prefix)
            files.append(fn)
        status.add_files(files)
        return files

    def sample(s_in, s, z, tag):
        x = mm.get_x_noisy(mm.get_x_start(B), 0)
        with torch.no_grad():
            for i in range(N):
                if STOP:
                    raise KeyboardInterrupt
                x0 = mm.get_x_0_hat_from_x_noisy_batched(x, start_index=i, s_inputs=s_in, s_trunk=s, z_trunk=z)
                x = mm.get_x_t_from_x_0_hat(x, x0, i, i + 1)
                if i < N - 1:
                    x = mm.get_x_noisy(x, i + 1)
                if i % 10 == 0:
                    status.update(stage=f"sampling {tag}", step=i, steps=N)
        _, rep = rloss(x)
        return x, rep

    metrics = {"settings": S}
    if S["baseline"]:
        xb, rep = sample(s_inputs0, s0, z0, "baseline")
        metrics["baseline"] = rep
        save(xb, "baseline")
        status.update(baseline=rep)

    # ---- IT-opt -------------------------------------------------------------
    s_in = s_inputs0.unsqueeze(0).expand(B, -1, -1).clone()
    s = s0.unsqueeze(0).expand(B, -1, -1).clone().requires_grad_(True)
    z = z0.unsqueeze(0).expand(B, -1, -1, -1).clone().requires_grad_(True)
    lr = float(S["lr"])
    gnorm = float(S["max_grad_norm"])
    history = []
    t_opt = time.time()
    try:
        for outer in range(int(S["outer"])):
            x = mm.get_x_noisy(mm.get_x_start(B), 0)
            for i in range(N):
                if STOP:
                    raise KeyboardInterrupt
                if i >= int(S["guide_from_step"]):
                    opt = torch.optim.Adam([s, z], lr=lr)  # re-created per step, as in it_opt
                    opt.zero_grad()
                    x0g = mm.denoise_net_batched(
                        x_noisy=x.detach(), t_hat_noise_level=mm.get_t_hat(i).reshape([1]).repeat(B),
                        input_feature_dict=mm.input_feature_dict, s_inputs=s_in, s_trunk=s, z_trunk=z,
                        chunk_size=mm.chunk_size, inplace_safe=False)
                    main, rep = rloss(x0g, i)
                    ls, lz = anchor(s, z)
                    (main + ls + lz).backward()
                    torch.nn.utils.clip_grad_norm_([s], gnorm)
                    torch.nn.utils.clip_grad_norm_([z], gnorm)
                    opt.step()
                    last_loss = float(main.detach())
                else:
                    rep, last_loss = None, None
                with torch.no_grad():
                    x0 = mm.get_x_0_hat_from_x_noisy_batched(x, start_index=i, s_inputs=s_in,
                                                             s_trunk=s.detach(), z_trunk=z.detach())
                    x = mm.get_x_t_from_x_0_hat(x, x0, i, i + 1)
                    if i < N - 1:
                        x = mm.get_x_noisy(x, i + 1)
                if i % 5 == 0 or i == N - 1:
                    done = outer * N + i + 1
                    el = time.time() - t_opt
                    status.update(stage="optimizing", outer=outer, outers=int(S["outer"]), step=i, steps=N,
                                  loss=last_loss, restraints=rep, eta_s=el / done * (int(S["outer"]) * N - done),
                                  anchor_s=float(ls.detach()) if rep else None, anchor_z=float(lz.detach()) if rep else None)
            _, rep = rloss(x.detach())
            history.append({"outer": outer, "restraints": rep, "loss": sum(r["loss"] for r in rep)})
            save(x, f"iter{outer:03d}")
            status.update(history=history)
    except KeyboardInterrupt:
        status.update(stage="stopping", note="stopped by user; sampling from current embeddings")

    # ---- final unguided samples from optimised embeddings -----------------
    STOP = False
    xf, rep = sample(s_in, s.detach(), z.detach(), "final")
    metrics["final"] = rep
    metrics["history"] = history
    save(xf, "final")
    torch.save({"s_inputs": s_in.cpu(), "s": s.detach().cpu(), "z": z.detach().cpu()},
               os.path.join(out_dir, "optimized_embeddings.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    status.update(stage="done", final=rep, t_total=time.time() - status.state["t_start"])


if __name__ == "__main__":
    main()
