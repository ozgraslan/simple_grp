# Simple GRP

A simple implementation of Octo-like policy with PyTorch. 

## Current implementation supports
- Train and validation with lerobot/libero_10_image dataset.
- Train and validation with lerobot/libero_10_image dataset.
- Evaluate the model in libero simulation.
- Blockwise causal attention between (goal text, obs image), (obs state), (action).
- No history.
- Action chunking. Predicts given number of actions but uses the first predicted action in env eval.
- T5 used for computing goal text embeddings.
  - Goal text embeddings are precomputed, and T5 is not trained.
  - No projection layer for text embeddings.
- Observation image is patchified anhttps://huggingface.co/datasets/openvla/modified_libero_rldsd projected using two layer norms and one linear layer.
- Learnable action token is used to predict actions.
  - Processed action token is projected to action using one linear layer.

## ToDo
- [ ] Implement argument parsing.
- [ ] Integrate model to lerobot (policy/config) repo.
- [ ] Add vision backbone support (dino, clip, etc).
