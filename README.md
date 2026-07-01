# Generative Inference

**Recognition gradients as a substrate for Bayesian perception**
Tahereh Toosi and Kenneth D. Miller · Columbia University

[![Paper](https://img.shields.io/badge/paper-bioRxiv-b31b1b)](https://www.biorxiv.org/content/10.1101/2025.10.21.683535v2) [![Demo](https://img.shields.io/badge/demo-HuggingFace-yellow)](https://huggingface.co/spaces/ttoosi/GenerativeInferenceDemo)

*Under review at Nature Neuroscience. A shorter version appeared at the NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning.*

---

## TL;DR

Score-based generative models train a separate network, a denoiser or diffusion model, to estimate the score $\nabla_x \log p(x)$, the quantity needed for sampling and for solving inverse problems. This work shows that an adversarially robust image classifier already carries an estimate of that quantity in its own loss gradient, and that running the same frozen network backward at test time turns it into a perceptual inference engine.

No second model, no retraining, no architecture change. By Bayes' rule applied to the classification loss, the input gradient decomposes into a class-conditional score and the natural-image prior score; robust training makes the recognition gradient a usable estimate of the prior score. We validate this alignment in closed form in a 2D Gaussian mixture, then apply the resulting algorithm to a robust ImageNet classifier. Under one set of inference parameters it reproduces illusory contours, figure-ground segregation, bistable percepts, and several brightness and color illusions, and its iteration dynamics track the delayed, feedback-driven responses recorded in early visual cortex.

## Why it connects to current ML

- **Test-time inference on a frozen network.** Inference is iterative gradient ascent on a fixed classifier, with no fine-tuning or added modules. This is a form of test-time computation for perception, and it works across ResNet-18/50, Wide-ResNet, VGG-16, and ViT-Large.
- **A score estimate you already have.** The same forward network that performs recognition supplies, through its loss gradient, the score directions that drive inference. One network covers both recognition and generation, rather than training a dedicated denoiser or diffusion model.
- **A use for adversarial robustness.** Robust training is what makes the recognition gradient align with the data score (near-unit cosine similarity in the closed-form 2D case; standard training does not align). Robustness here is an enabling property, not only a defense.
- **Discriminative and generative in one set of weights.** The forward pass recognizes; the same feedback pathway, run at test time, performs generative inference. The two modes are selected by the inference objective, not by different networks.

For contrast, solving the same Kanizsa completion as a classical inverse problem with Diffusion Posterior Sampling and a pretrained ImageNet diffusion prior recovered the percept in 2 of 75 configurations. A broad image-scale prior plus a likelihood term does not concentrate on the perceptual solution; the recognition-gradient ascent does.

## Method in one paragraph

Perception is cast as posterior ascent, $x_{t+1} = x_t + \alpha \nabla_x \log p(x_\text{sensory} \mid x_t) + \beta \nabla_x \log p(x_t)$. The prior-score term is supplied by the classifier's own gradient under a *Decrease Uncertainty* objective (ascend the loss toward the least-confident class), or its stochastic, representation-space variant for intermediate layers. Robust training's local smoothness is the condition under which the class-conditional term drops out and the recognition gradient approximates the prior score. Adding Langevin noise turns the deterministic update into posterior sampling for multi-stable stimuli.

## Repository contents

- `generative_inference/` — the core update rule and both prior-score estimators (Algorithms 1 and 2 in the paper)
- `stimuli/` — generation scripts for every stimulus (Kanizsa, Rubin face-vase, texture-defined figure-ground, neon color spreading, Ehrenstein, Cornsweet, confetti, checker-shadow, grouping displays)
- `experiments/` — reproduction scripts for each main-text figure
- `controls/` — Diffusion Posterior Sampling, PCN, and BLT-VS comparison pipelines
- `toy_2d/` — closed-form Gaussian-mixture validation of gradient-to-score alignment
- `notebooks/` — a short walkthrough matching the demo

> Adjust the paths above to match the actual layout before publishing.

## Reproducing the main result

```bash
# environment
pip install -r requirements.txt

# fetch the robust ImageNet checkpoint (Madry robustness library, ResNet-50, L2 eps=3.0)
python scripts/download_checkpoints.py

# reproduce the Kanizsa contour emergence (Figure 2a)
python experiments/kanizsa.py --config configs/unified.yaml
```

Trained checkpoints and the parametric stimulus sets are the ones listed in the paper's Table 1; the unified inference configuration is in `configs/unified.yaml`.

## Citation

```bibtex
@article{toosi2025generative,
  title   = {Generative inference unifies feedback processing for learning and perception in natural and artificial vision},
  author  = {Toosi, Tahereh and Miller, Kenneth D.},
  journal = {bioRxiv},
  year    = {2025},
  note    = {Under review at Nature Neuroscience}
}
```

## Related work from this line

- Toosi & Issa (2023), *Brain-like flexible visual inference by harnessing feedback-feedforward alignment*, NeurIPS. [[code]](https://github.com/toosi/Feedback_Feedforward_Alignment)
- Toosi (2025), *Interpretability at the network level: Prior-Guided Drift Diffusion for neural circuit analysis*, NeurIPS Mechanistic Interpretability Workshop.
