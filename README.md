# CompleteSCION
Official code for the paper **Learning Rate Transfer and Feature Learning Across Depth for Constrained Spectral Optimizers: Complete SCION**

This repository contains the code used to reproduce all experiments from the paper, including:

- Experiments with SCION, Unconstrained SCION, and the spectral clipping variant: Projected SCION
- Depth-wise learning rate transfer
- Width-wise learning rate transfer
- Feature learning vs. linearization analyses
- Alignment and cancellation studies

Experiments are provided on:

- nanoGPT / FineWeb
- DeiT / ImageNet
- MLP-Mixer / CIFAR-10

## Repository Organization

The repository is organized by experiments and contains several largely independent experimental projects.

| Directory | Details |
|------------|---------|
| `alignment_and_cancellation/` | Studies of alignment, residual cancellation, and update scaling across depth and width (CIFAR-10) |
| `deit/` | Depth-transfer experiments on DeiT (ImageNet) |
| `mlpmixer/` | Depth-transfer experiments on MLP-Mixer (CIFAR-10) |
| `nanogpt/` | Depth-transfer experiments on nanoGPT (FineWeb) |
| `modded_/` | Depth-transfer experiments on nanoGPT experiments with explicit Z-scaling analysis (FineWeb)|
| `relative_distance_linearization/` | Feature learning and distance-from-linearization analyses (MNIST) |
| `shallow_nanogpt/` | Width-transfer studies on shallow GPT models (Shakespeare-char) |



The experiments are self-contained and executable independently. Each experimental folder  ,contains:
- training scripts
- optimizer implementations
- plotting utilities
- configuration files (when applicable)
