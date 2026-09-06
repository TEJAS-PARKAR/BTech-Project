"""
Multi-Task Risk Prediction Network
=====================================
Takes the assembly relationship feature vector and produces:
  1. Interference Risk  ∈ [0, 1]   (regression)
  2. Clearance Risk     ∈ [0, 1]   (regression)
  3. Alignment Risk     ∈ [0, 1]   (regression)
  4. Fit Failure Prob.  ∈ [0, 1]   (regression)
  5. Risk Level ∈ {Low, Medium, High}  (3-class classification)

Also contains the full end-to-end AssemblyFitRiskModel that chains:
  PointNet++ Backbone → Assembly Relation Module → Risk Prediction Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from models.backbone import PointNet2Backbone
from models.assembly_relation import AssemblyRelationModule


class RiskPredictionHead(nn.Module):
    """
    Multi-task prediction network with shared trunk and dedicated heads.
    """

    def __init__(self, input_dim: int = 512, hidden_dims: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Shared trunk
        trunk_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            trunk_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.shared_trunk = nn.Sequential(*trunk_layers)

        last_dim = hidden_dims[-1]

        # Individual regression heads (each outputs a single scalar)
        self.interference_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.clearance_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.alignment_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.failure_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Classification head: Low (0) / Medium (1) / High (2)
        self.classification_head = nn.Sequential(
            nn.Linear(last_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # logits for 3 classes
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, input_dim) assembly relationship features

        Returns:
            dict with:
                'interference_risk': (B,) ∈ [0, 1]
                'clearance_risk':    (B,) ∈ [0, 1]
                'alignment_risk':    (B,) ∈ [0, 1]
                'failure_prob':      (B,) ∈ [0, 1]
                'risk_logits':       (B, 3) raw logits for classification
                'risk_level':        (B,) predicted class {0, 1, 2}
        """
        shared = self.shared_trunk(features)  # (B, last_dim)

        interference = self.interference_head(shared).squeeze(-1)
        clearance = self.clearance_head(shared).squeeze(-1)
        alignment = self.alignment_head(shared).squeeze(-1)
        failure = self.failure_head(shared).squeeze(-1)

        risk_logits = self.classification_head(shared)  # (B, 3)
        risk_level = torch.argmax(risk_logits, dim=-1)  # (B,)

        return {
            "interference_risk": interference,
            "clearance_risk": clearance,
            "alignment_risk": alignment,
            "failure_prob": failure,
            "risk_logits": risk_logits,
            "risk_level": risk_level,
        }


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for assembly fit risk prediction.

    L = λ₁·MSE(interference) + λ₂·MSE(clearance) + λ₃·MSE(alignment)
      + λ₄·BCE(failure)      + λ₅·CE(risk_level)
    """

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        if weights is None:
            weights = {
                "interference": 1.0,
                "clearance": 1.0,
                "alignment": 1.0,
                "failure": 1.5,
                "classification": 2.0,
            }
        self.weights = weights
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: dict from RiskPredictionHead
            targets:     dict from DataLoader labels

        Returns:
            total_loss: scalar tensor
            loss_dict:  breakdown of individual losses (for logging)
        """
        l_interf = self.mse(predictions["interference_risk"],
                            targets["interference_risk"])
        l_clear = self.mse(predictions["clearance_risk"],
                           targets["clearance_risk"])
        l_align = self.mse(predictions["alignment_risk"],
                           targets["alignment_risk"])
        l_fail = self.bce(predictions["failure_prob"],
                          targets["failure_prob"])
        l_cls = self.ce(predictions["risk_logits"],
                        targets["risk_level"])

        total = (self.weights["interference"] * l_interf +
                 self.weights["clearance"] * l_clear +
                 self.weights["alignment"] * l_align +
                 self.weights["failure"] * l_fail +
                 self.weights["classification"] * l_cls)

        loss_dict = {
            "interference_loss": l_interf.item(),
            "clearance_loss": l_clear.item(),
            "alignment_loss": l_align.item(),
            "failure_loss": l_fail.item(),
            "classification_loss": l_cls.item(),
            "total_loss": total.item(),
        }
        return total, loss_dict


# ──────────────────────────────────────────────
# Full End-to-End Model
# ──────────────────────────────────────────────

class AssemblyFitRiskModel(nn.Module):
    """
    Complete Assembly Fit Risk Prediction Model.

    Pipeline:
        Point Cloud (B, N, 7)
            → PointNet++ Backbone
            → Assembly Relationship Module
            → Multi-Task Risk Prediction Heads
            → {Interference, Clearance, Alignment, Failure, Risk Level}
    """

    def __init__(self, part_embed_dim: int = 16, max_parts: int = 8,
                 num_heads: int = 4, relation_hidden_dim: int = 256,
                 risk_hidden_dims: list = None, dropout: float = 0.3):
        super().__init__()

        # PointNet++ Feature Backbone
        self.backbone = PointNet2Backbone(
            part_embed_dim=part_embed_dim,
            max_parts=max_parts,
        )

        # Assembly Relationship Analysis
        self.relation_module = AssemblyRelationModule(
            backbone_feat_dim=640,
            num_heads=num_heads,
            hidden_dim=relation_hidden_dim,
            dropout=dropout,
        )

        # Risk Prediction Heads
        self.risk_head = RiskPredictionHead(
            input_dim=self.relation_module.output_dim,
            hidden_dims=risk_hidden_dims or [256, 128],
            dropout=dropout,
        )

    def forward(self, point_cloud: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            point_cloud: (B, N, 7) – [x, y, z, nx, ny, nz, part_id]

        Returns:
            predictions dict with all risk values and classifications
        """
        # Step 1: PointNet++ feature extraction
        backbone_out = self.backbone(point_cloud)

        # Step 2: Assembly relationship analysis
        relation_features = self.relation_module(backbone_out, point_cloud)

        # Step 3: Risk prediction
        predictions = self.risk_head(relation_features)

        return predictions

    def get_num_parameters(self) -> Dict[str, int]:
        """Return parameter counts per module."""
        counts = {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "relation_module": sum(p.numel() for p in self.relation_module.parameters()),
            "risk_head": sum(p.numel() for p in self.risk_head.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


# Risk level label mapping
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


if __name__ == "__main__":
    # Full model shape verification
    model = AssemblyFitRiskModel(
        part_embed_dim=16, max_parts=8,
        num_heads=4, relation_hidden_dim=256,
        risk_hidden_dims=[256, 128], dropout=0.3,
    )

    dummy = torch.randn(2, 2048, 7)
    dummy[:, :, 6] = torch.randint(0, 2, (2, 2048)).float()

    preds = model(dummy)
    print("=== Assembly Fit Risk Model Output ===")
    for k, v in preds.items():
        print(f"  {k}: {v.shape}")

    params = model.get_num_parameters()
    print(f"\n=== Parameter Counts ===")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
