# Parameter-Efficient Few-Step Distillation of Diffusion Models via LoRA

**Computer Vision (Prof. Irene Amerini) — Project 5: Lightweight Diffusion Models — Accelerating Training/Inference for Resource-Constrained Environments**

CIFAR-10 · PyTorch · single-GPU (Google Colab / Kaggle T4)

---

## 1. Overview

Denoising Diffusion Probabilistic Models (DDPMs) generate high-quality images but are slow at inference because they require hundreds/thousands of sequential denoising steps. This project reduces that cost and studies the resulting **quality-vs-speed trade-off**.

We train a standard DDPM baseline on CIFAR-10 and then integrate our methodological novelty:

> **Parameter-efficient few-step distillation via LoRA.** We freeze the trained DDPM (the *teacher*) and train only small **LoRA adapters** injected into the attention layers (the *student*) with a **trajectory-matching distillation objective** (progressive-distillation style). The student learns to reproduce, in very few steps (K = 8, 4, 2), what the teacher produces with many DDIM sub-steps — compensating the discretization error that makes naive few-step sampling collapse.

Because only the adapters are trained (a small fraction of the weights) and they can be merged into the base weights at inference, the method adds **essentially zero inference overhead** while providing large speed-ups over the full-step baseline.

**Key result:** at a matched step budget, the distilled student clearly beats plain DDIM — e.g. at **K = 4, FID 103.3 → 68.3 (Δ +34.9)** — and a naive-eps LoRA control confirms that the gain comes from the *distillation objective*, not from the extra parameters.

---

## 2. Repository structure

```
.
├── README.md                         # this file
├── notebook/
│   └── CV_Project5_final.ipynb       # full pipeline (executed, with outputs)
├── report/
│   ├── REPORT.md                     # written report / slides backbone
│   └── slides.pdf                    # project presentation
└── results/
    ├── results_full_ablation.csv     # all methods × metrics
    ├── paired_comparison.csv         # matched-K DDIM vs LoRA + causal control
    ├── run_config.json               # resolved hyper-parameters of the run
    ├── requirements_pinned.txt       # exact environment versions
    └── samples/                      # qualitative sample grids (PNG)

```

The notebook follows the structure requested by the course: **Imports → Globals → Utils → Data → Network → Train → Evaluation**.

---

## 3. Dataset

**CIFAR-10** (50k train / 10k test, 32×32 RGB), used under its standard license.
It is downloaded automatically by `torchvision.datasets.CIFAR10`; on Kaggle an offline fallback path is included. No manual download is required. FID/IS/Precision/Recall are computed against the CIFAR-10 **test** split via `torch-fidelity`.

---

## 4. How to run

### Environment
Single GPU (tested on Kaggle T4). Exact versions are pinned in `results/requirements_pinned.txt`:

```
python 3.12 · torch 2.10.0+cu128 · torchvision 0.25.0+cu128
numpy 2.0.2 · pandas 2.3.3 · torch-fidelity 0.4.0
```

`torch-fidelity` is installed inside the notebook (`pip install torch-fidelity`).

### Steps
1. Open `notebook/CV_Project5_final.ipynb` in Kaggle/Colab and enable the GPU accelerator.
2. In the **Globals** cell, choose a run profile:

   ```python
   RUN_PROFILE = "kaggle_strong_10k"
   ```

3. Run all cells top-to-bottom. Training resumes from `ddpm_latest.pt` if present, so an interrupted session can continue.
4. Artifacts (CSVs, `run_config.json`, `requirements_pinned.txt`, sample grids) are written to the working directory and zipped by the final cell.

### Run profiles (speed vs completeness)

| Profile | epochs | base_channels | FID images | Purpose |
|---|---|---|---|---|
| `kaggle_debug`     | 2  | 48 | 512   | End-to-end sanity check (~min) |
| `kaggle_quick`     | 10 | 48 | 2048  | Fast partial run |
| `kaggle_improved_t4` | 50 | 48 | 10000 | Stronger T4-friendly run |
| `kaggle_final_10k` | 50 | 64 | 10000 | Full run |
| **`kaggle_strong_10k`** | **80** | **64** | **10000** | **Profile used for the reported results** |

> Tip: run `kaggle_debug` first to validate the pipeline end-to-end before launching a long run.

### Reproducibility
Fixed seed (`SEED = 42`), `cudnn.deterministic = True`, `torch.use_deterministic_algorithms(True)`, and a seeded `torch.Generator` so that **all methods denoise the same initial noise bank** (paired comparison). The resolved config and pinned versions are exported with the results.

---

## 5. Results snapshot (profile `kaggle_strong_10k`, FID on 10k images)

**Baseline — quality/speed vs number of DDIM steps.** Fewer steps = faster but lower quality; note how **recall (diversity) collapses** long before precision does.

| Sampler | Steps | FID ↓ | IS ↑ | Precision | Recall | s/img | Speed-up vs DDPM-1000 |
|---|---|---|---|---|---|---|---|
| DDPM (ref, 1k imgs) | 1000 | 67.3* | 6.26 | 0.847 | 0.381 | 0.678 | 1× |
| DDIM | 100 | 29.33 | 7.23 | 0.751 | 0.326 | 0.053 | 12.8× |
| DDIM | 50  | 29.79 | 7.32 | 0.777 | 0.306 | 0.027 | 25.2× |
| DDIM | 20  | 32.43 | 7.32 | 0.834 | 0.243 | 0.011 | 60.5× |
| DDIM | 10  | 40.89 | 7.02 | 0.893 | 0.141 | 0.006 | 112.9× |
| DDIM | 8   | 47.12 | 6.83 | 0.903 | 0.102 | 0.005 | 136.5× |
| DDIM | 4   | 103.28| 5.29 | 0.853 | 0.015 | 0.003 | 242.4× |
| DDIM | 2   | 188.84| 3.73 | 0.401 | 0.001 | 0.002 | ~360× |
| DDIM | 1   | 325.11| 1.56 | 0.005 | 0.000 | 0.001 | ~510× |

\* DDPM-1000 FID is on 1,000 images (noisier) — used only for speed-up factors, not for quality comparison.

**Novelty — matched step budget K: plain DDIM-K vs our distilled LoRA-K (same noise, same weights).**

| K | DDIM FID | Distilled FID | Δ FID | DDIM Recall | Distilled Recall | Inference time ratio |
|---|---|---|---|---|---|---|
| 8 | 47.12  | **38.24** | **+8.9** | 0.102 | 0.159 | 1.01× |
| 4 | 103.28 | **68.34** | **+34.9** | 0.015 | 0.051 | 1.00× |
| 2 | 188.84 | **159.12**| **+29.7** | 0.001 | 0.002 | 1.01× |
| 1 | 325.11 | 326.57 | −1.5 | 0.000 | 0.000 | 1.00× |

**Causal control — distillation objective vs naive eps-loss LoRA (same rank, same params).**

| K | Naive-eps LoRA Δ FID | Distillation Δ FID |
|---|---|---|
| 8 | +0.96 | **+8.9** |
| 4 | −3.51 (worse) | **+34.9** |

The naive control gives ≈0 or negative gain, while distillation gives a large gain → the improvement is caused by the **objective**, not by the extra LoRA parameters.

Full numbers (rank ablation, state-source ablation, KID) are in `results/`.

---

## 6. Method in one paragraph

Standard DDPM (Ho et al., 2020) with an attention U-Net, linear β-schedule, T = 1000, ε-prediction MSE loss, trained with EMA and mixed precision. Fast sampling via deterministic DDIM (Song et al., 2021). The novelty distills the DDPM into a few-step student by training only LoRA adapters (Hu et al., 2021) on the attention projections with a progressive-distillation-style trajectory-matching loss (Salimans & Ho, 2022): for a random interval `t → t'` of the student schedule, the teacher refines with several small DDIM sub-steps and the student must match it in one step.

---

## 7. References
- J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
- T. Salimans, J. Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022.
- E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021.
- T. Kynkäänniemi et al. *Improved Precision and Recall Metric for Assessing Generative Models.* NeurIPS 2019.

---

## 8. Authors
Claudia Giuseppini,
Benedetta Conti,
Fabio Falconi

Computer Vision, Sapienza University of Rome, 2026.
