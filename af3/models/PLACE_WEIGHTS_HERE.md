# AlphaFold 3 weights go here

The AlphaFold 3 checkpoint has been **removed from this archive**: the weights
are access-gated by Google DeepMind and may not be redistributed.

## How to obtain

Request access via the AlphaFold 3 repository's *"Obtaining Model Parameters"*
section: <https://github.com/google-deepmind/alphafold3>

## Where to place it

Put the checkpoint in **this directory**:

```
af3/models/af3.bin.zst        (or af3.bin — the runner reads either)
```

Runs point at this directory with `--model_dir=.../af3/models`.
See [`../README.md`](../README.md) → section **"Model weights"**.
