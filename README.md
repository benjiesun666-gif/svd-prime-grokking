# Emergent Low-Rank Structure in Neural Networks Trained on Prime Numbers
https://zenodo.org/badge/DOI/10.5281/zenodo.18639667.svg

## ⚠️ Important Update (February 2026):

This repository now contains the refined analysis using Singular Value Decomposition (SVD). While the symmetrized weights of random initializations naturally follow Gaussian Orthogonal Ensemble (GOE) statistics (a known fact from random matrix theory), training can significantly alter the original (non‑symmetrized) weight matrices. We therefore quantify this change via three SVD‑based metrics: effective rank, the ratio of the largest to mean singular value, and the Kolmogorov–Smirnov distance to a random Gaussian baseline. This approach cleanly separates the innate randomness of initialization from the structure induced by learning.

## 📄 Paper

- **Title**: Emergent Low‑Rank Structure in Neural Networks Trained on Prime Numbers
- **Author**: Yunshan Yan (Lanzhou Jiaotong University)
- **Contact**: 20253607238@stu.lzjtu.edu.cn

### Current Version (v3.0 – SVD Analysis)

- **Repository**: This GitHub (February 2026)
- **Key Insight**: Randomly initialized weights already exhibit GOE after symmetrization; the real signal lies in the deviation from randomness of the original weight matrices.
- **Status**: Manuscript under review (revision)

### Previous Versions

- **v2.0 (GOE analysis)**: Archived in this repository (`branch goe-analysis`).
- **v1.0 (GUE analysis)**: Archived on Zenodo: `10.5281/zenodo.18377560`

## 🔬 Abstract

We investigate how training data shapes the statistical structure of Transformer weight matrices. Using singular value decomposition (SVD), we quantify the deviation from randomness via three metrics: effective rank, the ratio of the largest to mean singular value, and the Kolmogorov–Smirnov distance to the singular value spectrum of a random Gaussian matrix. We find that randomly initialized weights already exhibit a near‑full‑rank structure (effective rank ≈ 256, max/mean ratio ≈ 2–3). Training on prime gaps to predict the next gap produces two sharply distinct outcomes: successful learning (Plans A and B) induces a strongly low‑rank structure (effective rank 27–44, max/mean ratio 106–128, KS distance ~0.9), while a model that fails to learn (Plan C) remains close to the random baseline (effective rank 199, max/mean ratio 2.8, KS distance 0.81). Extending the analysis to 11 pretrained language models reveals a wide spectrum of structural richness: some models (e.g., TinyLlama) exhibit low‑rank characteristics similar to successful prime models, whereas others stay near‑random. This establishes SVD‑based metrics as powerful tools for quantifying the information content encoded in neural network weights.

## 🧪 Experimental Configurations

We designed three complementary experiments to isolate the source of the observed structural changes. The training task is the same for all: predict the gap between consecutive primes given the prime index.

### Experiment Plans

| Experiment | Embedding Type     | Batch Strategy          | Effective Batch | Description              |
|-----------|--------------------|-------------------------|----------------|--------------------------|
| Plan A    | Learnable          | Gradient Accumulation   | 2048           | Optimal training (baseline) |
| Plan B    | Learnable          | Standard                | 256            | Test batch size effect   |
| Plan C    | Sinusoidal (Fixed) | Gradient Accumulation   | 2048           | Test embedding flexibility |

### Configuration Details

#### Plan A: Optimal Training

```python
LEARNABLE_EMBEDDING = True
USE_GRADIENT_ACCUMULATION = True
PHYSICAL_BATCH_SIZE = 128
ACCUMULATION_STEPS = 16
# Effective batch = 128 × 16 = 2048
```

#### Plan B: Small Batch Control

```python
LEARNABLE_EMBEDDING = True
USE_GRADIENT_ACCUMULATION = False
PHYSICAL_BATCH_SIZE = 256
# Effective batch = 256 (no accumulation)
```

#### Plan C: Fixed Encoding Control

```python
LEARNABLE_EMBEDDING = False  # Uses sinusoidal positional encoding
USE_GRADIENT_ACCUMULATION = True
PHYSICAL_BATCH_SIZE = 128
ACCUMULATION_STEPS = 16
# Effective batch = 128 × 16 = 2048
```

## 📊 Key Results

### Prime‑Trained Models

| Model  | Effective Rank ↓ | Max/Mean Ratio ↑ | KS Distance to Random ↑ | Training Outcome  |
|--------|------------------|------------------|--------------------------|-------------------|
| Plan A | 44.13            | 106.16           | 0.9258                   | ✅ Best MAE (0.39) |
| Plan B | 27.30            | 127.62           | 0.9062                   | ⚠️ Moderate (0.74) |
| Plan C | 198.82           | 2.79             | 0.8086                   | ❌ Failed (3.97)   |
| Random (baseline)\* | 254.5 ± 0.9 | 2.35 ± 0.09 | – | – |

\*Random baseline computed from three untrained instances of the same architecture.

**Key Finding**: Successful learning induces a dramatic low‑rank compression, while a model that fails to learn remains statistically near‑random, despite prolonged training.

### Control Experiments (11 Pretrained Language Models)

| Model Family | Model Name         | Effective Rank | Max/Mean Ratio | KS Distance to Random |
|--------------|--------------------|----------------|----------------|------------------------|
| Qwen 2.5     | Qwen2.5‑0.5B       | 69.25          | 42.46          | 0.8945                 |
| Qwen 2.5     | Qwen2.5‑1.5B       | 171.44         | 8.83           | 0.9062                 |
| Qwen 2.5     | Qwen2.5‑3B         | 199.02         | 3.68           | 0.9531                 |
| GPT‑2        | gpt2               | 156.76         | 4.35           | 0.7656                 |
| GPT‑2        | gpt2‑medium        | 96.60          | 12.02          | 0.7578                 |
| GPT‑2        | gpt2‑large         | 188.80         | 3.04           | 0.8828                 |
| TinyLlama    | TinyLlama‑1.1B‑Chat| 38.41          | 43.25          | 0.9648                 |
| OLMo         | OLMo‑1B‑hf         | 175.73         | 5.56           | 0.9805                 |
| Pythia       | pythia‑410m        | 189.60         | 4.77           | 0.9570                 |
| Pythia       | pythia‑1b          | 187.68         | 11.69          | 0.9844                 |
| BLOOM        | bloom‑560m         | 192.49         | 3.01           | 0.9688                 |

**Interpretation**: Language models span a wide spectrum from near‑random (e.g., Qwen2.5‑3B) to strongly structured (e.g., TinyLlama), indicating that the compressibility of training data varies greatly.

## 📂 Repository Structure

```text
├── train_prime_gap.py               # Main training script (Plans A, B, C)
├── svd_prime_analysis.py            # SVD analysis of prime‑trained models
├── svd_language_models_analysis.py  # SVD analysis of 11 pretrained language models
├── svd_random_baseline.py           # Generates random Gaussian baseline
├── Plan A.txt / Plan B.txt / Plan C.txt   # Complete training logs
├── transformer_svd_analysis.json    # SVD results for language models
├── 论文图片/                          # All figures (300 DPI PNG)
│   ├── Plan_A_prediction.png
│   ├── Plan_B_prediction.png
│   ├── Plan_C_prediction.png
│   ├── Plan_A_svd_spectrum.png
│   ├── Plan_B_svd_spectrum.png
│   ├── Plan_C_svd_spectrum.png
│   ├── loss_curves_all_plans.png
│   ├── transformer_batch_analysis.png
│   ├── Plan_A_goe_spectrum.png       # GOE verification (Supplementary)
│   ├── Plan_B_goe_spectrum.png
│   └── Plan_C_goe_spectrum.png
└── ...
```

## 🚀 Quick Start

### Requirements

```bash
pip install torch numpy matplotlib sympy scikit-learn scipy huggingface_hub safetensors
```

### Training (Prime Gap Models)

```bash
# Edit train_prime_gap.py to select Plan A, B, or C, then run:
python train_prime_gap.py
```

### SVD Analysis

```bash
# Analyze prime‑trained models
python svd_prime_analysis.py

# Analyze language models (downloads weights from Hugging Face)
python svd_language_models_analysis.py

# Generate random baseline
python svd_random_baseline.py
```

### Reproduce Figures

Figures are generated automatically by the analysis scripts. For the summary plot of language models, run `svd_language_models_analysis.py`, which saves `transformer_batch_analysis.png`.

## 📦 Model Weights

Due to file size (994.8 MB each), model checkpoints are hosted on Google Drive:

**Download Links:**
- **Plan A.pt** (Learnable + Large Batch): [Download](https://drive.google.com/file/d/173NjtTL1M71oDjaxwYrt0xgAyhKi6sVu/view?usp=drive_link)
- **Plan B.pt** (Learnable + Small Batch): [Download](https://drive.google.com/file/d/1FSXscn4fEi9DYSYM0lN-zwU7QvKdlBmn/view?usp=drive_link)
- **Plan C.pt** (Sinusoidal + Large Batch): [Download](https://drive.google.com/file/d/1KSRG4cpMFqXDwddZu0FxqipkSB7aIGc1/view?usp=drive_link)

## 🔬 Reproducibility

All experiments were conducted on a Kaggle P100 GPU (16GB VRAM). Training logs include complete hyperparameter evolution history, ensuring full reproducibility.

### Training Details

- **Dataset**: First 1,000,000 primes (2 to 15,485,863)
- **Task**: Predict prime gap \(p_{n+1} - p_n\) from index \(n\)
- **Model**: Transformer Encoder (6 layers, 256 hidden dim, 8 heads)
- **Optimizer**: AdamW (lr=1e-4, weight decay=0.01)
- **Duration**: ~11 hours per experiment (160–325 epochs)

## 📧 Contact

- **Name**: Yunshan Yan  
- **Affiliation**: Lanzhou Jiaotong University  
- **Email**: 20253607238@stu.lzjtu.edu.cn

## 📜 Citation

If you use this code or findings, please cite:

### For the current SVD‑based analysis (recommended):

```bibtex
@misc{yan2026emergence_svd,
  title={Emergent Low-Rank Structure in Neural Networks Trained on Prime Numbers},
  author={Yan, Yunshan},
  year={2026},
  howpublished={GitHub repository},
  note={SVD analysis version; previous GOE and GUE versions archived},
  url={https://github.com/benjiesun666-gif/svd-prime-grokking}
}
```

### For the original GUE‑based version:

```bibtex
@misc{yan2026emergence_gue,
  title={Emergent Gaussian Unitary Ensemble Statistics in Neural Networks Trained on Prime Numbers},
  author={Yan, Yunshan},
  year={2026},
  howpublished={Zenodo},
  doi={10.5281/zenodo.18377560},
  url={https://doi.org/10.5281/zenodo.18377560}
}
```

## ⚖️ License

This project is licensed under the MIT License – see the `LICENSE` file for details.

---

Last updated: February 2026

