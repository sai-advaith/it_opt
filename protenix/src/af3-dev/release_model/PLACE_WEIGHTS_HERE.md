# Protenix model checkpoint goes here

The checkpoint has been **removed from this archive** to keep it lightweight.
It is publicly downloadable.

## How to obtain

```bash
# run from this directory
wget https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.2.0.pt
```

## Where to place it

The resulting file must live at:

```
protenix/src/af3-dev/release_model/model_v0.2.0.pt
```

This path is referenced by `pipeline_configurations/*.yaml` as
`model_checkpoint_path`. See [`../../../README.md`](../../../README.md) →
Installation, step 2.
