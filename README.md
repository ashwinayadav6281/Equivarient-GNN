# Equivariant-GNN for Elastic Tensor Prediction

This repository contains a complete pipeline for predicting the full 6x6 elastic stiffness tensor ($C_{ij}$) of crystalline materials directly from their atomic structures (CIF files) using **Equivariant Graph Neural Networks (E3NN)**.

## 🚀 Project Overview & Milestones Achieved

Predicting full elastic tensors is a notoriously difficult task due to the high dimensionality (21 independent components), class imbalance (many components are exactly zero due to crystal symmetries), and severe outliers in DFT-computed datasets. 

Throughout this project, we have completely overhauled the data acquisition, model architecture, and training pipelines to solve these challenges:

* **Massively Accelerated Data Pipeline:** Replaced sequential API scraping with direct bulk JSON processing. Successfully merged 32k+ JARVIS compounds with 12k+ Materials Project (MP) compounds into a unified, formatted dataset.
* **Physics-Informed Outlier Filtering:** Implemented Voigt-Reuss-Hill (VRH) averaging schemes to calculate Bulk ($K$) and Shear ($G$) moduli. Used these bounds to detect and remove ~2,300 unphysical DFT outliers (singular matrices causing moduli in the billions of GPa), resulting in a pristine dataset of **~45,000 compounds**.
* **Metal vs. Non-Metal Classification:** Categorized the entire 45k dataset into Metals and Non-Metals using local JARVIS metadata and the original pre-trained CGCNN classification model for the obfuscated MP subset (Resulting in ~31.7k Metals and ~13.9k Non-metals).
* **High-Efficiency "Lean" Model:** Redesigned the E3NN architecture to reduce parameters from 4.5M to ~424K. This drastically accelerated CPU training time from 75 mins/epoch down to **~10 mins/epoch** without sacrificing representational capacity.
* **Radial Basis Encodings:** Integrated a 10-feature Gaussian `soft_one_hot_linspace` radial basis to give the model a much stronger understanding of interatomic distances.
* **Symmetry-Aware Loss Function:** Engineered a custom loss function that applies a `1.5x` penalty to true-zero components (encouraging the model to learn crystal symmetries) and a `3x` boost to non-zero off-diagonal components (combating the heavy bias towards large diagonal $C_{11}, C_{22}, C_{33}$ values).
* **Remote Server Deployment:** Fully integrated automated scripts for deploying code, syncing datasets, managing `nohup` training runs, and fetching logs from a headless, GPU-less remote server.

---

## 📂 Repository Structure

### Core Model & Training
* **`equivariant_model.py`**: Defines the `ImprovedEquivariantElasticNet`. Uses `e3nn` to handle $O(3)$ symmetries and invariant multi-layer perceptrons for the readout phase.
* **`train_equivariant_server.py`**: The primary training script designed for CPU servers. Includes gradient accumulation, `CosineAnnealingWarmRestarts`, and the custom symmetry-aware L1 Loss function.
* **`predict_tensor_components.py`**: Evaluates the trained model on test data, un-normalizes the predictions, and outputs side-by-side comparisons of True vs Predicted components.

### Data Acquisition & Preprocessing
* **`download_jarvis.py` / `convert_jarvis.py`**: Rapidly downloads and extracts the bulk JARVIS DFT dataset, mapping 9-component Voigt notations to full 36-component matrices.
* **`gen_preprocess.py`**: Standardizes the inputs, handles graph generation radius cutoffs, and formats the dataset for the PyTorch Geometric DataLoader.
* **`tensor_utils.py`**: Contains crucial physics utilities, including the extraction of exact Voigt, Reuss, and Hill moduli from arbitrary 6x6 tensors.
* **`clean_and_plot_vrh.py`**: Cleans the dataset of physical outliers ($K$ or $G$ > 1000 GPa or < -50 GPa) and generates density hexbin plots.
* **`classify_metals.py` / `merge_classification.py`**: Scripts utilizing JARVIS metadata and CGCNN models to classify materials into Metals/Non-Metals for stratified analysis.

### Visualization & Analysis
* **`Plots/bulk_vs_shear_moduli_clean.png`**: Hexbin density plot of the VRH Bulk vs Shear moduli of the clean dataset, featuring the Pugh's ratio ductile/brittle dividing line.

---

## 📊 Dataset Breakdown

After processing and filtering, the dataset consists of **45,613** clean crystalline structures:
* **JARVIS Subset:** 32,585 compounds (22,625 Metals, 9,960 Non-Metals)
* **Materials Project Subset:** 13,028 compounds (9,090 Metals, 3,938 Non-Metals)

A lookup file mapping every `material_id` to its respective class can be found at `dataset_equivariant/metal_classification.json`.

---

## ⚙️ Usage

**1. Training the Model**
To start training from scratch on the server using the cleaned dataset:
```bash
source venv/bin/activate
nohup python3 train_equivariant_server.py --epochs 100 > equivariant_training.log 2>&1 &
```

**2. Evaluating Predictions**
To run inference on the test set and see the breakdown of individual component errors:
```bash
python3 predict_tensor_components.py
```

*Note: Large datasets (CIF directories, pre-trained `.pth.tar` weights) are intentionally excluded from this repository via `.gitignore` to maintain repository performance and size limits.*
