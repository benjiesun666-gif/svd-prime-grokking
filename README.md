# Emergent GOE Statistics in Neural Networks Trained on Prime Numbers

⚠️ **Important Update**:  
This repository contains the refined analysis using **Gaussian Orthogonal Ensemble (GOE)**, which is physically correct for real-valued neural network weight matrices.

## 📄 Paper

**Title**: Emergent Gaussian Orthogonal Ensemble Statistics in Neural Networks Trained on Prime Numbers  
**Author**: Yunshan Yan (Lanzhou Jiaotong University)  
**Contact**: 20253607238@stu.lzjtu.edu.cn  

### Current Version (v2.0 - Refined GOE Analysis)
- **Repository**: This GitHub (February 2026)
- **Key Improvement**: 75% better statistical fit using GOE framework (KS distance ~0.05 vs. ~0.17 in v1.0)
- **Physical Rationale**: GOE is the theoretically correct reference for real-valued neural network weights
- **Status**: Manuscript under review at Science Advances

### Original Version (v1.0 - Archived)
- **Zenodo DOI**: [10.5281/zenodo.18377560](https://doi.org/10.5281/zenodo.18377560)
- **Published**: January 26, 2026
- **Note**: Original analysis archived for reference; this repository contains the refined GOE-based approach

## 🔬 Abstract

We investigate the statistical properties of weight matrices in Transformer models trained to predict prime number gaps. Remarkably, we observe spontaneous emergence of Gaussian Orthogonal Ensemble (GOE) statistics—a hallmark of time-reversal symmetric chaotic systems—in the eigenvalue spacings of symmetrized weight matrices. This phenomenon persists across diverse training configurations and, crucially, is **independent of prediction accuracy**. Control experiments on 13 general-purpose language models show Poisson statistics (71.4% Poisson, 0% pure GOE), suggesting that prime number complexity induces chaotic dynamics in neural networks.

**Physical Justification**: Since neural network weight matrices are real-valued and symmetric, GOE is the theoretically correct reference class from Random Matrix Theory, providing significantly better statistical fit than GUE (complex Hermitian matrices).

## 🧪 Experimental Configurations

We designed three complementary experiments to isolate the source of GOE statistics:

| Experiment | `LEARNABLE_EMBEDDING` | `USE_GRADIENT_ACCUMULATION` | Effective Batch | Description |
|------------|----------------------|----------------------------|-----------------|-------------|
| **Plan A** | `True` | `True` | 2048 | Learnable embedding + Large batch (optimal baseline) |
| **Plan B** | `True` | `False` | 256 | Learnable embedding + Small batch (test batch size effect) |
| **Plan C** | `False` | `True` | 2048 | Sinusoidal embedding + Large batch (test embedding flexibility) |

### Configuration Details

**Plan A: Optimal Training**
```python
LEARNABLE_EMBEDDING = True
USE_GRADIENT_ACCUMULATION = True
PHYSICAL_BATCH_SIZE = 128
ACCUMULATION_STEPS = 16
# Effective batch = 128 × 16 = 2048
```

**Plan B: Small Batch Control**
```python
LEARNABLE_EMBEDDING = True
USE_GRADIENT_ACCUMULATION = False
PHYSICAL_BATCH_SIZE = 256
# Effective batch = 256 (no accumulation)
```

**Plan C: Fixed Encoding Control**
```python
LEARNABLE_EMBEDDING = False  # Uses sinusoidal positional encoding
USE_GRADIENT_ACCUMULATION = True
PHYSICAL_BATCH_SIZE = 128
ACCUMULATION_STEPS = 16
# Effective batch = 128 × 16 = 2048
```

## 📊 Key Results

| Experiment | Embedding | Batch Size | MAE | GOE Distance | Verdict | Training Outcome |
|-----------|-----------|------------|-----|--------------|---------|------------------|
| **Plan A** | Learnable | Large (2048) | 0.39 | **0.049** | **GOE** | ✅ Best performance |
| **Plan B** | Learnable | Small (256) | 0.74 | **0.050** | **GOE** | ⚠️ Moderate performance |
| **Plan C** | Sinusoidal | Large (2048) | 3.97 | **0.056** | **GOE** | ❌ **Failed to learn** |

**Key Finding**: GOE emergence is **independent of prediction accuracy** and **unique to prime numbers**. The refined GOE analysis shows 70%+ improvement in statistical fit (KS distance) compared to previous GUE analysis.

### Control Experiments (13 Transformer Models)
- Qwen2.5 (0.5B, 1.5B, 3B): **Poisson**
- GPT-2 (small, medium, large): **Poisson**
- TinyLlama-1.1B: **Poisson**
- OLMo-1B: **Poisson**
- Total: 71.4% Poisson, 28.6% Mixed, 0% Pure GOE

**Interpretation**: General language models show Poisson statistics (uncorrelated eigenvalues), while prime-trained networks exhibit GOE statistics (strong level repulsion), indicating that prime number complexity induces chaotic dynamics.

## 📂 Repository Structure

```
├── prime_goe_emergence.py          # Main training script (1769 lines)
├── batch_generate_figures.py       # Generate prediction & GOE analysis plots
├── test_batch_models.py           # Transformer control experiments (13 models)
├── verify_weights.py              # Weight analysis and GOE verification script
├── Plan A.txt / Plan B.txt / Plan C.txt   # Complete training logs
├── transformer_spectra_analysis.json      # Control experiment results
└── figures/                         # All figures (300 DPI PNG)
    ├── Plan_A_prediction.png
    ├── Plan_A_verification.png      # GOE/GUE/Poisson comparison
    ├── Plan_B_prediction.png
    ├── Plan_B_verification.png
    ├── Plan_C_prediction.png
    ├── Plan_C_verification.png
    ├── loss_curves_all_plans.png
    └── transformer_batch_analysis.png
```

## 🚀 Quick Start

### Requirements
```bash
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
sympy>=1.12
scikit-learn>=1.3.0
scipy>=1.11.0
```

### Installation
```bash
pip install torch numpy matplotlib sympy scikit-learn scipy
```

### Training

**Run Plan A (Optimal Configuration):**
```bash
# Edit prime_goe_emergence.py:
LEARNABLE_EMBEDDING = True
USE_GRADIENT_ACCUMULATION = True

# Run training (requires GPU, tested on Kaggle P100 16GB VRAM)
python prime_goe_emergence.py
```

**Run Plan B (Small Batch):**
```bash
# Edit prime_goe_emergence.py:
LEARNABLE_EMBEDDING = True
USE_GRADIENT_ACCUMULATION = False

python prime_goe_emergence.py
```

**Run Plan C (Sinusoidal Encoding):**
```bash
# Edit prime_goe_emergence.py:
LEARNABLE_EMBEDDING = False
USE_GRADIENT_ACCUMULATION = True

python prime_goe_emergence.py
```

### Generate Figures
```bash
python batch_generate_figures.py
```

### Control Experiments
```bash
python test_batch_models.py
```

## 📦 Model Weights

Due to file size (994.8 MB each), model checkpoints are hosted on Google Drive:

**Download Links:**
- **Plan A.pt** (Learnable + Large Batch): [Download](https://drive.google.com/file/d/173NjtTL1M71oDjaxwYrt0xgAyhKi6sVu/view?usp=drive_link)
- **Plan B.pt** (Learnable + Small Batch): [Download](https://drive.google.com/file/d/1FSXscn4fEi9DYSYM0lN-zwU7QvKdlBmn/view?usp=drive_link)
- **Plan C.pt** (Sinusoidal + Large Batch): [Download](https://drive.google.com/file/d/1KSRG4cpMFqXDwddZu0FxqipkSB7aIGc1/view?usp=drive_link)

## 🔬 Reproducibility

All experiments were conducted on **Kaggle P100 GPU (16GB VRAM)**. Training logs include complete hyperparameter evolution history, ensuring full reproducibility.

### Training Details
- **Dataset**: First 1,000,000 primes (2 to 15,485,863)
- **Task**: Predict prime gap $p_{n+1} - p_n$ from index $n$
- **Model**: Transformer Encoder (6 layers, 256 hidden dim, 8 heads)
- **Optimizer**: AdamW (lr=1e-4, weight decay=0.01)
- **Duration**: ~11 hours per experiment (160-325 epochs)

## 📧 Contact

**Yunshan Yan**  
Lanzhou Jiaotong University  
Email: 20253607238@stu.lzjtu.edu.cn

## 📜 Citation

If you use this code or findings, please cite:

### For the refined GOE analysis (recommended):
```bibtex
@misc{yan2026emergent_goe,
  title={Emergent Gaussian Orthogonal Ensemble Statistics in Neural Networks Trained on Prime Numbers},
  author={Yan, Yunshan},
  year={2026},
  howpublished={GitHub repository (v2.0 - GOE Analysis)},
  note={Refined analysis with 75\% improved statistical fit. Original version archived at Zenodo (DOI: 10.5281/zenodo.18377560)},
  url={https://github.com/[YOUR_USERNAME]/prime-goe-emergence}
}
```

### For the original archived version:
```bibtex
@misc{yan2026emergent_original,
  title={Emergent Gaussian Unitary Ensemble Statistics in Neural Networks Trained on Prime Numbers},
  author={Yan, Yunshan},
  year={2026},
  howpublished={Zenodo},
  doi={10.5281/zenodo.18377560},
  url={https://doi.org/10.5281/zenodo.18377560}
}
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Last updated: January 2026*

