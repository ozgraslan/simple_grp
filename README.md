# Simple GRP

A simple implementation of Octo-like policy with PyTorch. 
Some of the code is adapted from [Lucidrain](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py),
and from Mini-Grp course code.

## Current implementation supports
- Train and validation with lerobot/libero_10_image dataset.
- Train and validation with lerobot/libero_10_image dataset.
- Evaluate the model in libero simulation.
- Blockwise causal attention between (goal text, obs image), (obs state), (action).
- No history.
- Action chunking. Predicts given number of actions and uses the given number actions (open loop) in env eval.
- T5 used for computing goal text embeddings.
  - Goal text embeddings are precomputed, and T5 is not trained.
  - No projection layer for text embeddings.
- Learnable action token is used to predict actions.
  - Processed action token is projected to action using one linear layer.
- There are multiple versions to process image observations differently:
  - simple_grp.py : Observation image is patchified anhttps://huggingface.co/datasets/openvla/modified_libero_rldsd projected using two layer norms and one linear layer.
  - dinoreg_grp_tsd.py : Observation images are processed using a pretrained dinov2 with registers model.
    - dataset_dinoreg.py : Used to preprocess dataset with dinov2 with registers and save it separately.
  - ctrlo_grp_tsd: Observation images are processed using a pretrained ctrl-o model. 
    - dataset_ctrlo.py : Used to preprocess dataset with ctrl-o and save it separately.

## ToDo
- [ ] Implement argument parsing.
- [ ] Integrate model to lerobot (policy/config) repo.
