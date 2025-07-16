# 3D Coronary Tree Reconstruction from X-ray Projections

A PyTorch-based implementation for reconstructing 3D coronary tree structures from multi-view X-ray projections, with a focus on solving gradient vanishing issues in projection-based optimization.

## Table of Contents
- [Installation and Setup](#installation-and-setup)
- [Code Architecture](#code-architecture)
- [Critical Bug Fixes and Solutions](#critical-bug-fixes-and-solutions)
- [Usage](#usage)
- [Results](#results)

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for processing high-resolution volumes

### Environment Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd 3d-coronary-reconstruction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Data preparation:**
   - Place your CCTA data (`.label.nii.gz` files) in `/path/to/CCTA_raw/1-200/`
   - Update data paths in the configuration files if necessary

### Running the Code

The reconstruction pipeline consists of two main steps:

#### Step 1: Generate Optimal Projections
```bash
python optimal_projection_generator.py
```

This script will:
- Process all CCTA 3D coronary tree samples from the raw data directory
- Calculate optimal resolution parameters based on geometric constraints
- Generate both continuous and binary projections
- Save projection data to `/path/to/CCTA_optimal_proj/`
- **Automatically generate `config.py`** with optimal parameters

#### Step 2: Run 3D Reconstruction
```bash
python main.py
```

This script will:
- Load the generated projection data
- Perform 3D reconstruction using the fixed optimization pipeline
- Generate comprehensive evaluation metrics and visualizations
- Save results to `reconstruction_results_fixed/`

---

## Code Architecture

### Core Components

#### 1. `optimal_projection_generator.py`
**Purpose:** Processes raw CCTA data and generates optimized projection images for reconstruction.

**Workflow:**
1. **Load Binary CCTA 3D Coronary Tree:** Reads `.label.nii.gz` files containing segmented coronary tree structures
2. **3D Resolution Adjustment:** 
   - **Problem:** Original resolution insufficient for high-quality 2D projections, causing projection artifacts
   - **Solution:** Calculate optimal voxel spacing using geometric constraints from X-ray projection theory
3. **Trilinear Interpolation:**
   - **Why needed:** Resolution change requires resampling the volume to find new voxel values at interpolated positions
   - **Process:** Uses PyTorch's `F.interpolate()` with trilinear mode for smooth volume resampling
4. **Threshold-based Binarization:** Apply 0.5 threshold to maintain binary vessel structure
5. **Continuous Projection Generation:** 
   - **Key Innovation:** Generates both continuous `[0, 67]` and binary `{0, 1}` projections
   - **Continuous projections preserve ray intensity information** (number of voxels each ray passes through)
   - **Binary projections maintain compatibility** with traditional visualization

#### 2. `projector.py`
**Purpose:** Implements matrix-based X-ray projection with multiple output modes.

**Key Features:**
- **`project_volume_continuous()`:** Preserves ray intensity information `[0, max_voxels]`
- **`project_volume_binary()`:** Traditional binary output `{0, 1}` for visualization
- **`project_volume_normalized()`:** Normalized continuous output `[0, 1]`
- **Sparse matrix implementation** for memory efficiency with large volumes

#### 3. `reconstruction.py`
**Purpose:** Core reconstruction algorithm with gradient-aware optimization.

**Key Components:**
- **`FixedDifferentiableProjection`:** Custom autograd function ensuring proper gradient flow
- **91-sample initialization strategy:** Uses real coronary tree data as initialization
- **Self-consistency testing:** Validates gradient flow with 91→91 reconstruction test
- **Comprehensive metrics calculation:** Dice, IoU, clDice, reError, reMSE, Chamfer L2

#### 4. `data_loader.py`
**Purpose:** Handles loading and preprocessing of projection data with multiple formats.

**Features:**
- **Automatic format detection:** Prioritizes continuous projections, falls back to binary
- **Mode selection:** `'continuous'`, `'binary'`, or `'auto'`
- **Data validation:** Ensures projection consistency and proper tensor formatting

#### 5. `utils.py`
**Purpose:** Mathematical utilities for optimal resolution calculation.

**Core Function:**
- **`calculate_voxel_spacing()`:** Implements geometric constraints from projection theory to determine optimal voxel spacing based on SOD, SID, pixel spacing, and projection angles

#### 6. `visualization.py`
**Purpose:** Comprehensive visualization suite for results analysis.

**Capabilities:**
- **3D volume rendering** with multiple viewing angles
- **Projection comparison** visualizations
- **Loss curve plotting** with convergence analysis
- **Multi-threshold isosurface** generation

#### 7. `main.py`
**Purpose:** Orchestrates the complete reconstruction pipeline.

**Workflow:**
1. Data structure verification and projection loading
2. Individual sample reconstruction with comprehensive analysis
3. Forward projection verification (reprojection consistency check)
4. 3D visualization generation and GT comparison
5. Metrics calculation and summary table generation

#### 8. `config.py` (Auto-generated)
**Purpose:** Centralized configuration with optimal parameters.

**Contains:**
- Geometric parameters (SOD, SID, pixel spacing)
- Optimal volume resolution and voxel spacing
- Projection angles optimized for maximum separation
- Data paths and reconstruction parameters

---

## Critical Bug Fixes and Solutions

### Problem 1: Poor Initialization Strategy

**Issue:** Random and zero initializations yielded poor reconstruction quality.

**Solution:** **Withdrew real 3D coronary tree from CTA data for initialization.**
- Used sample 91's actual coronary tree structure as initialization
- Significantly enhanced reconstruction convergence and quality
- Provides anatomically plausible starting point for optimization

### Problem 2: Volume-Projection Scaling Mismatch

**Issue:** When using sample 91 as initialization, `P_vec(X)` didn't match sample 91's target projections `I_k`.

**Root Cause:** Inconsistent voxel coordinate scaling between projection generation and reconstruction phases.

**Original Problem in Code:**
```python
# In optimal_projection_generator.py
avg_voxel_size = np.mean(voxel_spacing)
projector.voxel_coords *= avg_voxel_size  # Applied scaling

# In reconstruction.py (original)
projector = MatrixXRayProjector(...)
# Missing: projector.voxel_coords *= avg_voxel_size  # Scaling not applied!
```

**Fix:** Ensured consistent scaling application in both generation and reconstruction phases:
```python
# In reconstruction.py (fixed)
projector = MatrixXRayProjector(...)
avg_voxel_size = np.mean(voxel_spacing)
projector.voxel_coords *= avg_voxel_size  # Now consistently applied
```

### Problem 3: Gradient Vanishing Due to `(proj_vec > 0)` Operation

**The Most Critical Issue:** Binary thresholding operation caused severe gradient vanishing and "fake convergence."

#### Problem Analysis

**Observed Symptoms:**
- Loss appeared to converge to small values (~1e-5)
- **Gradient norm consistently near zero** (~1e-8)
- Loss remained constant across iterations (no actual optimization)
- Self-consistency test (91→91 reconstruction) failed to achieve true zero loss

**Root Cause Investigation:**

1. **Information Dimension Mismatch:**
```python
# Problematic approach:
target_projections = load_continuous_projections()  # Range: [0, 67]
pred_projections = project_binary(X)                # Range: {0, 1}
loss = ||pred - target||²  # Comparing [0,67] vs {0,1}
```

2. **Gradient Flow Interruption:**
```python
def project_binary(X):
    proj_continuous = P @ X.flatten()     # Continuous: [0, 67]
    return (proj_continuous > 0).float()  # ⚠️ Step function with zero gradient!

# Mathematical issue:
∂(proj > 0)/∂proj = 0  (almost everywhere)
# Therefore: ∂loss/∂X = 0
```

3. **Numerical Dilution Effect:**
- Medical images are sparse (~1000 vessel pixels, ~212,000 background pixels)
- Large errors on vessel pixels get diluted by numerous zero-error background pixels
- Resulting loss appears small despite poor reconstruction quality

#### Solution Comparison

We evaluated multiple approaches to resolve the gradient vanishing issue:

| Approach | Gradient Preservation | Information Retention | Numerical Stability | Physical Meaning | Recommendation |
|----------|----------------------|----------------------|-------------------|------------------|----------------|
| **Continuous** | ✅ Perfect | ✅ Complete | ✅ Excellent | ✅ Accurate | ⭐⭐⭐⭐⭐ |
| Hard Clamp | ⚠️ Partial | ❌ Lost | ✅ Good | ⚠️ Approximate | ⭐⭐ |
| Soft Clamp | ✅ Present | ⚠️ Compressed | ✅ Good | ⚠️ Compressed | ⭐⭐⭐ |
| Sigmoid | ✅ Present | ⚠️ Compressed | ✅ Good | ⚠️ Compressed | ⭐⭐⭐ |
| Binary (Original) | ❌ Vanished | ❌ Severely Lost | ✅ Good | ❌ Incorrect | ❌ |

**Method Details:**

```python
# 1. Continuous (Final Solution)
def project_volume_continuous(self, X, alpha, beta):
    proj_vec = torch.sparse.mm(P, X.flatten().unsqueeze(-1))
    return proj_vec.reshape(self.image_shape)  # Preserve [0, 67] range

# 2. Hard Clamp (Tested)
proj_clamped = torch.clamp(proj_vec, min=0, max=1)  # Gradient=0 when proj_vec>1

# 3. Soft Clamp (Tested)
proj_soft = torch.tanh(proj_vec * 0.1)  # Smooth mapping with continuous gradients

# 4. Sigmoid (Tested)
proj_sigmoid = torch.sigmoid(proj_vec * 0.1)  # Smooth mapping to (0,1)
```

#### Final Solution: Continuous Projection Approach

**Implementation:**
1. **Data Generation:** Save both continuous `.pt` tensors and binary `.png` images
2. **Data Loading:** Prioritize loading continuous projections for reconstruction
3. **Optimization:** Use continuous projections throughout the optimization process
4. **Evaluation:** Maintain information consistency between target and prediction

**Results After Fix:**
- **Gradient norm:** >1000 (previously ~0)
- **Loss behavior:** Proper descent to true convergence (previously fake convergence)
- **Self-consistency test:** 91→91 reconstruction achieves loss <1e-6
- **Reconstruction quality:** Significantly improved across all metrics

**Why Continuous Projection Works:**
1. **Preserves Ray Intensity Information:** Each pixel value represents the number of voxels a ray passes through
2. **Maintains Gradient Flow:** Linear operation `P @ X` has well-defined gradients
3. **Information Consistency:** Both target and prediction operate in the same `[0, max_intensity]` space
4. **Physical Interpretation:** Ray intensity naturally corresponds to tissue density along the ray path

---

## Usage

### Quick Start
```bash
# Step 1: Generate projections and config
python optimal_projection_generator.py

# Step 2: Run reconstruction
python main.py
```

### Advanced Usage

**Custom data paths:**
```python
# Edit paths in optimal_projection_generator.py
self.label_root = "/your/path/to/CCTA_raw/1-200"
self.output_root = "/your/path/to/CCTA_optimal_proj"
```

**Reconstruction parameters:**
```python
# Edit config.py (auto-generated)
NUM_ITERATIONS = 5000  # Increase for better convergence
LEARNING_RATE = 0.01   # Adjust based on convergence behavior
```

### Output Structure
```
reconstruction_results_fixed/
├── detailed_metrics_table_fixed.csv
├── summary_metrics_table_fixed.csv
├── sample_*_volume_rendering_fixed.png
├── sample_*_comparison_fixed.png
├── forward_projection_verification_sample_*.png
├── reconstructed_binary_sample_*_fixed.npy
├── reconstructed_continuous_sample_*_fixed.npy
├── metrics_sample_*.json
└── metadata_sample_*_fixed.json
```

---

## Results

### Quantitative Metrics
The reconstruction pipeline evaluates performance using six standard metrics:
- **Dice Score:** Volumetric overlap measure
- **IoU (Jaccard Index):** Intersection over union
- **clDice:** Centerline-aware Dice score for tubular structures
- **reError:** Reconstruction error (L1 norm)
- **reMSE:** Reconstruction mean squared error
- **Chamfer L2:** Bidirectional point cloud distance

### Key Improvements After Bug Fixes
1. **Gradient Flow Restoration:** Gradient norms increased from ~0 to >1000
2. **True Convergence:** Loss properly descends to near-zero values
3. **Self-Consistency:** 91→91 reconstruction achieves expected perfect reconstruction
4. **Quality Enhancement:** Significant improvement across all evaluation metrics

### Validation
The pipeline includes comprehensive validation through:
- **Self-consistency testing** (reconstruct ground truth from its own projections)
- **Forward projection verification** (reproject reconstructed volumes)
- **Cross-sample analysis** with statistical significance testing

---
