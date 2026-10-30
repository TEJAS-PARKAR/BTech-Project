# Assembly Fit Risk Prediction using PointNet++

> Deep Learning-based Assembly Fit Risk Assessment from 3D Point Clouds

---

## 🎯 Problem Statement

Mechanical assemblies require precise dimensional fit between mating components. Traditional inspection methods (manual measurement, ICP registration) cannot predict assembly fit risk **before physical assembly**. Existing deep learning approaches focus on either:
- Point cloud registration only (ICP, DCP, PointNetLK)
- Object detection/classification (PointNet, Faster R-CNN)
- Joint prediction without tolerance analysis (JoinABLe)
- Disassembly direction only (AssemblyNet)

**None** of them predict **assembly fit quality** including interference, clearance, alignment, and failure risk.

## 🚀 Our Approach

We use **PointNet++** extended with a novel **Assembly Relationship Analysis Module** to directly predict fit risk from 7D point clouds:

```
Input: Point Cloud (x, y, z, nx, ny, nz, part_id)
    → Point Cloud Preprocessing
    → PointNet++ Feature Learning (Hierarchical MSG Set Abstraction)
    → Assembly Relationship Analysis (Cross-Part Attention + Interface Reasoning)
    → Multi-Task Risk Prediction
    → Interference Risk | Clearance Risk | Alignment Risk | Fit Failure Probability
    → Low / Medium / High Risk Classification
```

## 📁 Project Structure

```
btechproject/
├── data/
│   ├── dataset_generator.py      # Synthetic CAD assembly generator
│   ├── dataset.py                # PyTorch Dataset & DataLoader
│   └── samples/                  # Generated point cloud data
├── models/
│   ├── pointnet2_utils.py        # PointNet++ building blocks (FPS, Ball Query, SA)
│   ├── backbone.py               # PointNet++ backbone for 7D input
│   ├── assembly_relation.py      # Cross-Part Attention & Interface Analysis
│   └── risk_predictor.py         # Multi-task risk prediction + full model
├── training/
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation & metrics
│   └── config.yaml               # Hyperparameters
├── visualizer/
│   ├── app.py                    # Streamlit interactive dashboard
│   ├── cad_utils.py              # CAD mesh sampling and rigid auto-fit
│   └── plot_utils.py             # 3D Plotly visualization utilities
├── experiments/
│   └── benchmark_comparison.py   # Baseline comparison
├── requirements.txt
└── README.md
```

## 🔧 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python data/dataset_generator.py --num_samples 4000 --n_points 2048

# Train the model
python training/train.py --config training/config.yaml

# Evaluate on test set
python training/evaluate.py --checkpoint checkpoints/best_model.pth

# Run benchmark comparison
python experiments/benchmark_comparison.py

# Launch interactive dashboard
streamlit run visualizer/app.py
```

## Auto-Fit Two CAD Parts

The dashboard uses the same launch command above. In **Input Mode**, choose **Convert CAD Mesh**, upload Part A and Part B as separate files, choose 2048 or 4096 output points, then select **Auto-fit parts**. The app searches for up to three rigid placement candidates; choose one under **Suggested Placement** to view the combined point cloud and geometry indicators.

Supported mesh formats are STL, OBJ, PLY, GLB, and GLTF. Each uploaded file must contain one watertight solid. Native STEP and IGES files must first be exported to one of these tessellated mesh formats. Auto-fit changes only rotation and translation, not part dimensions. The solver is a heuristic, not a guarantee of functional mating; units are taken from the uploaded files and are not inferred. Overlap, contact, and gap values are approximate geometry indicators, not calibrated failure probabilities or tolerance checks.

The auto-fit implementation uses the existing `scipy` and `trimesh` dependencies in `requirements.txt`; no separate CAD fitting command or additional package installation is needed. After changing dependencies, install them with `pip install -r requirements.txt` and launch the dashboard with `streamlit run visualizer/app.py`.

## 🏗️ Architecture Details

### PointNet++ Backbone (7D Input Adaptation)
- **Part ID Embedding**: Learnable 16-dim embedding for categorical part identifiers
- **SA Layer 1 (MSG)**: 2048 → 512 points, radii [0.1, 0.2, 0.4], output 320-dim features
- **SA Layer 2 (MSG)**: 512 → 128 points, radii [0.2, 0.4, 0.8], output 640-dim features
- **Global SA**: 128 → 1 global vector (1024-dim)

### Assembly Relationship Module
- **Interface Extractor**: Nearest-neighbor distance field, proximity masks, normal compatibility
- **Cross-Part Attention**: Bidirectional multi-head attention (4 heads) between Part A ↔ Part B
- **Relational Fusion**: Concatenates attended part features + global feature + interface statistics

### Multi-Task Risk Head
| Output | Type | Activation | Loss |
|--------|------|-----------|------|
| Interference Risk | Regression [0,1] | Sigmoid | MSE |
| Clearance Risk | Regression [0,1] | Sigmoid | MSE |
| Alignment Risk | Regression [0,1] | Sigmoid | MSE |
| Failure Probability | Regression [0,1] | Sigmoid | BCE |
| Risk Level | 3-class | Softmax | Cross-Entropy |

### Combined Loss
$$\mathcal{L} = \lambda_1 \mathcal{L}_{interf} + \lambda_2 \mathcal{L}_{clear} + \lambda_3 \mathcal{L}_{align} + \lambda_4 \mathcal{L}_{fail} + \lambda_5 \mathcal{L}_{CE}$$

## 📊 Synthetic Dataset

Four assembly archetypes with parametric tolerance defects:

| Type | Description | Defect Simulation |
|------|-------------|-------------------|
| Shaft & Hole | Cylindrical fit | Radial expansion/contraction, tilt |
| Flange Joint | Bolted mating | Gap variation, angular offset |
| Peg-in-Hole | Prismatic insertion | Interference/clearance, eccentricity |
| Dovetail | Sliding joint | Width deviation, misalignment |


