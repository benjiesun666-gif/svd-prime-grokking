# Emergent Low-Rank Structure in Neural Networks Trained on Prime Numbers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18639667.svg)](https://doi.org/10.5281/zenodo.18639667)

## ⚠️ Important Update (March 2026)
  
Key updates:

- **SVD analysis** remains the core method (effective rank, max/mean ratio, KS distance to random baseline).
- **New symmetry analysis**: successful prime‑gap models (Plans A and B) develop a striking **symmetric double‑peak structure** in the row vectors of the first‑layer attention weights. This feature is quantified by inter‑cluster distance (13.66–19.74) and symmetry quality (60.09 for Plan A, 44.21 for Plan B), and is **completely absent** in the failed model (Plan C) and in a **controlled chaotic logistic map**.
- **Chaotic logistic map** serves as a counterexample: it learns perfectly (MAE = 0.028) but retains a high‑rank, non‑symmetric weight structure, demonstrating that learning fingerprints are data‑specific.

## 📄 Paper

- **Title**: Emergent Low‑Rank Structure in Neural Networks Trained on Prime Numbers
- **Author**: Yunshan Yan (Lanzhou Jiaotong University)
- **Contact**: 20253607238@stu.lzju.edu.cn

## 🔬 Abstract

We investigate how training data shapes the statistical structure of Transformer weight matrices. Using singular value decomposition (SVD) and row‑vector symmetry analysis, we quantify the deviation from randomness via three SVD metrics (effective rank, max/mean singular value ratio, Kolmogorov–Smirnov distance to a random Gaussian matrix) and two symmetry metrics (inter‑cluster distance, cluster balance). We find that randomly initialized weights exhibit a near‑full‑rank structure (effective rank ≈ 206, max/mean ratio ≈ 2–3). Training on prime gaps produces two sharply distinct outcomes: successful learning (Plans A and B) induces a strongly low‑rank structure (effective rank 27–44, max/mean ratio 106–128, KS ~0.9) **and** a symmetric double‑peak distribution of the row vectors (inter‑cluster distance 13.66–19.74, symmetry quality 60.09 for Plan A, 44.21 for Plan B). A model that fails to learn (Plan C) remains near‑random and shows no symmetry. As a control, a chaotic logistic map trained with the same architecture achieves excellent prediction (MAE = 0.028) but retains a high‑rank, non‑symmetric weight structure, demonstrating that the low‑rank + symmetry fingerprint is specific to prime‑gap learning. This work establishes a multi‑metric framework for probing the information content encoded in neural network weights.

## 🧪 Experimental Configurations

We designed three complementary experiments for the prime‑gap task. All models share the same Transformer architecture (6 layers, 256 hidden dim, 8 heads). The chaotic logistic map was trained with the Plan A configuration.

| Experiment | Embedding Type | Batch Strategy | Effective Batch | Description |
|------------|----------------|----------------|-----------------|-------------|
| Plan A | Learnable | Gradient Accumulation | 2048 | Optimal training (baseline) |
| Plan B | Learnable | Standard | 256 | Test batch size effect |
| Plan C | Sinusoidal (Fixed) | Gradient Accumulation | 2048 | Test embedding flexibility |

### Chaotic Map Control

- **Data**: Logistic map \(x_{n+1} = 3.8 x_n (1-x_n)\), 1,000,000 points, gaps as prediction target.
- **Configuration**: Identical to Plan A (learnable embedding, gradient accumulation, 300 epochs).

### IHC Ablation: Identity Hyper-Connection

To isolate the role of learnable cross‑flow mixing, we introduce the **Identity Hyper-Connection (IHC)** architecture. IHC preserves the multi‑stream design but fixes the mixing matrix to the identity, removing all learnable mixing while retaining the dynamic projection network (ϕ). This allows us to test whether the symmetric fingerprint requires adaptive mixing or can emerge from the multi‑stream topology alone. IHC models were trained on both prime gaps and the chaotic map using the Plan A hyperparameters (learnable embeddings, gradient accumulation, 300 epochs). An untrained IHC random baseline was also analyzed to establish the intrinsic compression of the architecture itself.

## 📊 Key Results

### Prime‑Trained Models and Chaotic Control

| Model | MAE | Effective Rank | Max/Mean Ratio | KS to Random | Inter‑cluster Distance | Cluster Balance | Symmetry Quality | Learning Outcome |
|-------|-----|----------------|----------------|--------------|------------------------|-----------------|------------------|------------------|
| Plan A | 0.3926 | 44.13 | 106.16 | 0.9258 | **13.66** | 0.992 | **60.09** | ✅ Success |
| Plan B | 0.7424 | 27.30 | 127.62 | 0.9062 | **19.74** | 0.977 | **44.21** | ✅ Success |
| Plan C | 3.9661 | 198.82 | 2.79 | 0.8086 | 0.54 | 0.656 | 1.75 | ❌ Failure |
| Chaotic (success) | 0.0283 | 203.34 | 2.52 | 0.8672 | 0.35 | 0.922 | 1.65 | ✅ Success |
| Chaotic (fail) | 0.4058 | 206.11 | 2.34 | 0.9453 | 0.14 | 0.953 | 1.65 | ❌ Failure |
| Random baseline* | – | 205.9 ± 0.2 | 2.34 ± 0.00 | 0.9414 ± 0.0000 | – | – | – | – |

*Random baseline computed from three untrained instances (see Section 3.4 in the paper).

**IHC Results**  
IHC models successfully learn both tasks (MAE = 0.19 for primes, 0.020 for chaos), but their weight structure is fundamentally different from the standard Transformer:
- **Effective rank** remains at the IHC random baseline (~51), unchanged by training.
- **Max/mean ratio** stays near the random level (~2.3), indicating no dominant direction emerges.
- **Symmetry quality** is uniformly low (1.77 for IHC Prime, 1.75 for IHC Chaos) in all layers, indistinguishable from failed models.
This confirms that the symmetric fingerprint and the surge in max/mean ratio require **learnable cross‑flow mixing** – a key property of the standard Transformer that is absent in IHC.

**Key Findings**:
1. **Successful prime models** exhibit both **low‑rank compression** and **symmetric row‑vector structure**.
2. **Failed prime model** and **successful chaotic model** do **not** share this fingerprint, showing that the observed structure is data‑specific.
3. The chaotic control demonstrates that a model can learn perfectly without any trace of low‑rank or symmetry.

## 📂 Repository Structure

```text
├── train_prime_gap.py               # Main training script for prime models
├── train_chaotic.py                 # Training script for chaotic logistic map
├── svd_prime_analysis.py            # SVD analysis for prime models
├── svd_chaotic_analysis.py          # SVD analysis for chaotic models
├── symmetry_analysis.py             # Row‑vector PCA + KMeans + symmetry metrics
├── svd_random_baseline.py           # Generate random Gaussian baseline
├── Plan A.txt / Plan B.txt / Plan C.txt  # Training logs
├── IHC_prime_gap_train.py            # IHC training on prime gaps
├── IHC_chaotic_train.py              # IHC training on chaotic map
├── random_baseline_ihc.py            # SVD analysis for untrained IHC models
├── layer_symmetry_analysis.py        # Layer‑wise symmetry quality (for all models)
├── 论文图片/                          # All figures (300 DPI PNG)
│   ├── Plan_A_prediction.png
│   ├── Plan_B_prediction.png
│   ├── Plan_C_prediction.png
│   ├── Plan_A_svd_spectrum.png
│   ├── Plan_B_svd_spectrum.png
│   ├── Plan_C_svd_spectrum.png
│   ├── loss_curves_all_plans.png
│   ├── chaos_svd_spectra.png         # SVD spectra of chaotic models
│   ├── chaos_prediction.png           # Prediction performance of chaotic models
│   ├── pca_clusters_all.png           # PCA row‑vector projections with clusters
│   └── ...
└── README.md
🚀 Quick Start
Requirements
bash
```
pip install torch numpy matplotlib sympy scikit-learn scipy
Training
bash
# Edit train_prime_gap.py to select Plan A, B, or C, then run:
```
python train_prime_gap.py
```
# Train chaotic model (choose plan A or C):
```
python train_chaotic.py --plan A   # or --plan C

python IHC_prime_gap_train.py          # IHC on prime gaps
python IHC_chaotic_train.py             # IHC on chaotic map
```
Analysis
bash
# Prime model SVD
```
python svd_prime_analysis.py
```
# Chaotic model SVD
```
python svd_chaotic_analysis.py
```
# Symmetry analysis (PCA + clustering)
```
python symmetry_analysis.py
```
# Generate random baseline
python svd_random_baseline.py
All scripts save figures in the 论文图片/ directory.

Download Links
**Prime Models (Standard Transformer)**  
- Plan A.pt (Learnable + Large Batch): [Download](https://drive.google.com/file/d/173NjtTL1M71oDjaxwYrt0xgAyhKi6sVu/view?usp=drive_link)  
- Plan B.pt (Learnable + Small Batch): [Download](https://drive.google.com/file/d/1FSXscn4fEi9DYSYM0lN-zwU7QvKdlBmn/view?usp=drive_link)  
- Plan C.pt (Sinusoidal + Large Batch): [Download](https://drive.google.com/file/d/1KSRG4cpMFqXDwddZu0FxqipkSB7aIGc1/view?usp=drive_link)  

**Chaotic Models (Standard Transformer)**  
- chaotic_success.pt (Plan A configuration): [Download](https://drive.google.com/file/d/1kGnxGi0tbhagNBVgJpMzHnmnRvespuyD/view?usp=sharing)  
- chaotic_fail.pt (Plan C configuration): [Download](https://drive.google.com/file/d/1Pmn_6lBvPGaAbqRAvrRFHQ31B0_jrbKo/view?usp=sharing)  

**IHC Models (Identity Hyper-Connection)**  
- IHC_prime.pt (trained on prime gaps): [Download](https://drive.google.com/file/d/1Hza4fbx6GPgS9YTMAqLyoZmsB0hhHEbz/view?usp=sharing)  
- IHC_chaos.pt (trained on chaotic map): [Download](https://drive.google.com/file/d/17aeTp6qeqF34z121iUjiRw43n6uN-kx0/view?usp=sharing)

##🔬 Reproducibility
All experiments were performed on a Kaggle P100 GPU (16GB VRAM). Training logs include hyperparameter histories; random seeds are fixed in analysis scripts for reproducibility.

##📧 Contact
Name: Yunshan Yan

Affiliation: Lanzhou Jiaotong University

Email: 20253607238@stu.lzju.edu.cn

##📜 Citation
If you use this code or findings, please cite the current version:

bibtex
@misc{yan2026emergence_svd_symmetry,
  title={Emergent Low-Rank Structure in Neural Networks Trained on Prime Numbers},
  author={Yan, Yunshan},
  year={2026},
  howpublished={GitHub repository},
  note={SVD + symmetry analysis version},
  url={https://github.com/benjiesun666-gif/svd-prime-grokking}
}
For previous versions, see the archived releases.

##⚖️ License
MIT License – see LICENSE file.

Last updated: April 2026
