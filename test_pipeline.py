"""Quick smoke test for the full pipeline."""
import sys
sys.path.insert(0, ".")

import torch
from data.dataset import create_dataloaders
from models.risk_predictor import AssemblyFitRiskModel, MultiTaskLoss

print("=" * 60)
print("SMOKE TEST: Assembly Fit Risk Prediction Pipeline")
print("=" * 60)

# 1. Test data loading
print("\n[1] Testing DataLoader...")
loaders = create_dataloaders("data/samples", batch_size=4, num_points=2048)
batch = next(iter(loaders["train"]))
pc, labels = batch
print(f"    Point cloud shape: {pc.shape}")
print(f"    Risk levels: {labels['risk_level'].tolist()}")
print(f"    Interference: {[f'{v:.3f}' for v in labels['interference_risk'].tolist()]}")
print(f"    Clearance:    {[f'{v:.3f}' for v in labels['clearance_risk'].tolist()]}")
print(f"    Alignment:    {[f'{v:.3f}' for v in labels['alignment_risk'].tolist()]}")

# 2. Test model forward pass
print("\n[2] Testing model forward pass...")
model = AssemblyFitRiskModel()
preds = model(pc)
for k, v in preds.items():
    print(f"    {k}: {v.shape}")

# 3. Test loss computation
print("\n[3] Testing multi-task loss...")
criterion = MultiTaskLoss()
loss, loss_dict = criterion(preds, labels)
print(f"    Total loss: {loss.item():.4f}")
for k, v in loss_dict.items():
    print(f"    {k}: {v:.4f}")

# 4. Test backward pass
print("\n[4] Testing backward pass...")
loss.backward()
print("    Gradient computation successful!")

# 5. Parameter count
params = model.get_num_parameters()
print(f"\n[5] Model parameters:")
for k, v in params.items():
    print(f"    {k}: {v:,}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED [OK]")
print("=" * 60)
