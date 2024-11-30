# DiT Transformer Implementation

This is an implementation of the Diffusion Transformer Architecture based on [this paper](https://arxiv.org/pdf/2212.09748) which was based on stable diffusion and ViT.

Some questions that I have and will attempt to answer is:

- How do attention layers 'beat' the learning/representational capacity of CNNs, which seem well suited for images (inductive bias of way animals see)?

- What is Xavier uniform intitialization, and why would it ever be used for non-linear activations over kaiming init, which actually preserves the variance?


- Why are we predicting noise and covaraince in our diffusion model? How do the mathematics allign with this objective; why can't we just pre-parameterize our posterior variance?

Answers:
1. Architectural complexity of CNNs vs. Transformers: how do we evaluate this (Raw GFLOPS vs. parallelization?)