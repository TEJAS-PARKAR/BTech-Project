"""
Assembly Relationship Analysis Module
=======================================
This module goes BEYOND standard PointNet++ (which treats the scene as a single
monolithic object) by explicitly modelling the geometric interactions between
distinct parts in a multi-component assembly.

Key components:
  1. Interface Point Extraction  – Identifies points near the mating boundary
  2. Cross-Part Attention        – Transformer attention between Part A ↔ Part B
  3. Relational Feature Fusion   – Encodes surface orientation compatibility,
                                    proximity statistics, and geometric interaction

This is the core novelty that overcomes the gaps in prior works (ICP, DCP,
AssemblyNet, JoinABLe) which cannot reason about fit quality between parts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.pointnet2_utils import square_distance


class CrossPartAttention(nn.Module):
    """
    Multi-Head Cross-Attention between features of two distinct parts.

    Given queries from Part A and keys/values from Part B (and vice versa),
    this module learns which regions of one part are relevant to the other,
    capturing mating surface correspondence and spatial compatibility.
    """

    def __init__(self, feat_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        assert feat_dim % num_heads == 0, \
            f"feat_dim ({feat_dim}) must be divisible by num_heads ({num_heads})"

        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_feats: torch.Tensor,
                context_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feats:   (B, N_q, D) features from one part
            context_feats: (B, N_c, D) features from the other part

        Returns:
            attended: (B, N_q, D) query features enriched with cross-part info
        """
        B, N_q, D = query_feats.shape
        H, d = self.num_heads, self.head_dim

        # Multi-head projections
        Q = self.q_proj(query_feats).view(B, N_q, H, d).transpose(1, 2)   # (B, H, N_q, d)
        K = self.k_proj(context_feats).view(B, -1, H, d).transpose(1, 2)  # (B, H, N_c, d)
        V = self.v_proj(context_feats).view(B, -1, H, d).transpose(1, 2)  # (B, H, N_c, d)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)  # (B, H, N_q, N_c)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, N_q, d)
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.out_proj(out)

        # Residual + LayerNorm
        x = self.norm1(query_feats + out)
        x = self.norm2(x + self.ffn(x))

        return x


class InterfaceExtractor(nn.Module):
    """
    Extract and analyse the mating interface between two parts.

    Computes:
    - Interface proximity mask (points near the boundary between parts)
    - Distance field statistics (min, mean, std of cross-part distances)
    - Normal compatibility (dot products between facing normals)
    """

    def __init__(self, interface_threshold: float = 0.15):
        super().__init__()
        self.threshold = interface_threshold

    def forward(self, xyz_a: torch.Tensor, xyz_b: torch.Tensor,
                normals_a: Optional[torch.Tensor] = None,
                normals_b: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            xyz_a:     (B, N_a, 3) coordinates of Part A points
            xyz_b:     (B, N_b, 3) coordinates of Part B points
            normals_a: (B, N_a, 3) surface normals of Part A (optional)
            normals_b: (B, N_b, 3) surface normals of Part B (optional)

        Returns:
            dict with interface features (all per-sample):
                'min_dist':           (B,) minimum cross-part distance
                'mean_interface_dist':(B,) mean distance at interface
                'std_interface_dist': (B,) std of interface distances
                'normal_alignment':   (B,) mean |n_a · n_b| at interface
                'interface_ratio_a':  (B,) fraction of Part A points at interface
                'interface_ratio_b':  (B,) fraction of Part B points at interface
                'interface_mask_a':   (B, N_a) boolean mask for Part A interface points
                'interface_mask_b':   (B, N_b) boolean mask for Part B interface points
        """
        # Pairwise squared distances: (B, N_a, N_b)
        cross_dist_sq = square_distance(xyz_a, xyz_b)
        cross_dist = torch.sqrt(cross_dist_sq + 1e-8)

        # Min distance from each Part A point to Part B
        min_dist_a, nn_idx_a = cross_dist.min(dim=-1)  # (B, N_a), (B, N_a)
        min_dist_b, nn_idx_b = cross_dist.min(dim=-2)  # (B, N_b), (B, N_b)

        # Interface masks
        interface_mask_a = min_dist_a < self.threshold  # (B, N_a)
        interface_mask_b = min_dist_b < self.threshold  # (B, N_b)

        B = xyz_a.shape[0]
        results = {}

        # Global min distance
        results["min_dist"] = min_dist_a.min(dim=-1)[0]  # (B,)

        # Interface distance statistics
        interface_dists = []
        mean_dists = torch.zeros(B, device=xyz_a.device)
        std_dists = torch.zeros(B, device=xyz_a.device)
        for b in range(B):
            mask_a = interface_mask_a[b]
            if mask_a.any():
                d = min_dist_a[b][mask_a]
                mean_dists[b] = d.mean()
                std_dists[b] = d.std() if d.numel() > 1 else 0.0
            else:
                mean_dists[b] = min_dist_a[b].mean()
                std_dists[b] = min_dist_a[b].std()

        results["mean_interface_dist"] = mean_dists
        results["std_interface_dist"] = std_dists

        # Interface ratios
        results["interface_ratio_a"] = interface_mask_a.float().mean(dim=-1)
        results["interface_ratio_b"] = interface_mask_b.float().mean(dim=-1)

        # Normal alignment at interface
        if normals_a is not None and normals_b is not None:
            # For each Part A interface point, get its nearest Part B normal
            batch_idx = torch.arange(B, device=xyz_a.device).unsqueeze(-1).expand_as(nn_idx_a)
            nn_normals_b = normals_b[batch_idx, nn_idx_a]  # (B, N_a, 3)
            # Dot product (facing normals should be anti-parallel → dot ≈ -1)
            dot_products = (normals_a * nn_normals_b).sum(dim=-1)  # (B, N_a)

            normal_alignment = torch.zeros(B, device=xyz_a.device)
            for b in range(B):
                mask_a = interface_mask_a[b]
                if mask_a.any():
                    normal_alignment[b] = dot_products[b][mask_a].mean()
                else:
                    normal_alignment[b] = dot_products[b].mean()
            results["normal_alignment"] = normal_alignment
        else:
            results["normal_alignment"] = torch.zeros(B, device=xyz_a.device)

        results["interface_mask_a"] = interface_mask_a
        results["interface_mask_b"] = interface_mask_b

        return results


class AssemblyRelationModule(nn.Module):
    """
    Full Assembly Relationship Analysis Module.

    Combines:
    1. Per-part feature separation from PointNet++ backbone
    2. Interface point extraction and distance field analysis
    3. Cross-part multi-head attention
    4. Relational feature fusion

    Produces a comprehensive assembly relationship vector.
    """

    def __init__(self, backbone_feat_dim: int = 640,
                 num_heads: int = 4, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        # Project backbone features to attention dimension
        self.feat_proj = nn.Sequential(
            nn.Linear(backbone_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Cross-part attention (bidirectional)
        self.cross_attn_a_to_b = CrossPartAttention(hidden_dim, num_heads, dropout)
        self.cross_attn_b_to_a = CrossPartAttention(hidden_dim, num_heads, dropout)

        # Interface extractor
        self.interface_extractor = InterfaceExtractor(interface_threshold=0.15)

        # Interface feature dimensions: 6 scalar features from interface analysis
        interface_feat_dim = 6

        # Fusion network
        # Input: attended_a_global + attended_b_global + global_feat + interface_features
        fusion_input_dim = hidden_dim * 2 + 1024 + interface_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.output_dim = 512

    def _separate_parts(self, xyz: torch.Tensor, features: torch.Tensor,
                        part_ids: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor]:
        """
        Separate point cloud features by part_id for cross-part analysis.

        For simplicity and batch compatibility, we use a binary mask approach:
        Part A = part_id == 0, Part B = part_id != 0.
        Points are zero-padded to ensure consistent tensor shapes.

        Returns:
            feat_a, feat_b: (B, N/2, D) features for each part (padded)
            xyz_a, xyz_b:   (B, N/2, 3) coordinates for each part (padded)
        """
        B, N, D = features.shape
        half_n = N // 2

        feat_a_list = []
        feat_b_list = []
        xyz_a_list = []
        xyz_b_list = []

        for b in range(B):
            mask_a = (part_ids[b] == 0)
            mask_b = ~mask_a

            fa = features[b][mask_a]
            fb = features[b][mask_b]
            xa = xyz[b][mask_a]
            xb = xyz[b][mask_b]

            # Pad or truncate to half_n
            if len(fa) >= half_n:
                fa = fa[:half_n]
                xa = xa[:half_n]
            else:
                pad = half_n - len(fa)
                fa = F.pad(fa, (0, 0, 0, pad))
                xa = F.pad(xa, (0, 0, 0, pad))

            if len(fb) >= half_n:
                fb = fb[:half_n]
                xb = xb[:half_n]
            else:
                pad = half_n - len(fb)
                fb = F.pad(fb, (0, 0, 0, pad))
                xb = F.pad(xb, (0, 0, 0, pad))

            feat_a_list.append(fa)
            feat_b_list.append(fb)
            xyz_a_list.append(xa)
            xyz_b_list.append(xb)

        feat_a = torch.stack(feat_a_list)  # (B, half_n, D)
        feat_b = torch.stack(feat_b_list)
        xyz_a = torch.stack(xyz_a_list)
        xyz_b = torch.stack(xyz_b_list)

        return feat_a, feat_b, xyz_a, xyz_b

    def forward(self, backbone_output: Dict[str, torch.Tensor],
                point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            backbone_output: dict from PointNet2Backbone with:
                - 'global_feat':  (B, 1024)
                - 'sa2_xyz':      (B, 128, 3)
                - 'sa2_features': (B, 128, 640)
                - 'part_ids':     (B, N)
            point_cloud: (B, N, 7) original input

        Returns:
            relation_features: (B, 512) assembly relationship feature vector
        """
        global_feat = backbone_output["global_feat"]      # (B, 1024)
        sa2_xyz = backbone_output["sa2_xyz"]               # (B, 128, 3)
        sa2_features = backbone_output["sa2_features"]     # (B, 128, 640)

        # We need part_ids at the SA2 level.
        # Approximate by nearest-neighbor assignment from original part_ids.
        original_xyz = point_cloud[:, :, :3]               # (B, N, 3)
        original_pids = point_cloud[:, :, 6]               # (B, N)
        # Find nearest original point for each SA2 centroid
        dists = square_distance(sa2_xyz, original_xyz)     # (B, 128, N)
        _, nn_idx = dists.min(dim=-1)                      # (B, 128)
        B = sa2_xyz.shape[0]
        batch_idx = torch.arange(B, device=sa2_xyz.device).unsqueeze(-1).expand_as(nn_idx)
        sa2_part_ids = original_pids[batch_idx, nn_idx]    # (B, 128)

        # Project features to attention dimension
        sa2_proj = self.feat_proj(sa2_features)  # (B, 128, hidden_dim)

        # Separate parts
        feat_a, feat_b, xyz_a, xyz_b = self._separate_parts(
            sa2_xyz, sa2_proj, sa2_part_ids)

        # Cross-part attention (bidirectional)
        attended_a = self.cross_attn_a_to_b(feat_a, feat_b)  # (B, N/2, hidden_dim)
        attended_b = self.cross_attn_b_to_a(feat_b, feat_a)  # (B, N/2, hidden_dim)

        # Aggregate attended features (global max pooling per part)
        global_a = attended_a.max(dim=1)[0]  # (B, hidden_dim)
        global_b = attended_b.max(dim=1)[0]  # (B, hidden_dim)

        # Interface analysis using original point cloud
        orig_xyz = point_cloud[:, :, :3]
        orig_normals = point_cloud[:, :, 3:6]
        orig_pids = point_cloud[:, :, 6]

        # Separate original points by part
        xyz_a_orig_list = []
        xyz_b_orig_list = []
        nrm_a_orig_list = []
        nrm_b_orig_list = []
        half_n_orig = point_cloud.shape[1] // 2

        for b in range(B):
            mask_a = (orig_pids[b] == 0)
            xa = orig_xyz[b][mask_a]
            xb = orig_xyz[b][~mask_a]
            na = orig_normals[b][mask_a]
            nb = orig_normals[b][~mask_a]

            # Pad/truncate
            for arr_list, arr, target_n in [
                (xyz_a_orig_list, xa, half_n_orig),
                (xyz_b_orig_list, xb, half_n_orig),
                (nrm_a_orig_list, na, half_n_orig),
                (nrm_b_orig_list, nb, half_n_orig),
            ]:
                if len(arr) >= target_n:
                    arr_list.append(arr[:target_n])
                else:
                    pad = target_n - len(arr)
                    arr_list.append(F.pad(arr, (0, 0, 0, pad)))

        xyz_a_orig = torch.stack(xyz_a_orig_list)
        xyz_b_orig = torch.stack(xyz_b_orig_list)
        nrm_a_orig = torch.stack(nrm_a_orig_list)
        nrm_b_orig = torch.stack(nrm_b_orig_list)

        interface_info = self.interface_extractor(
            xyz_a_orig, xyz_b_orig, nrm_a_orig, nrm_b_orig)

        # Build interface feature vector
        interface_feats = torch.stack([
            interface_info["min_dist"],
            interface_info["mean_interface_dist"],
            interface_info["std_interface_dist"],
            interface_info["normal_alignment"],
            interface_info["interface_ratio_a"],
            interface_info["interface_ratio_b"],
        ], dim=-1)  # (B, 6)

        # Fuse everything
        fused = torch.cat([global_a, global_b, global_feat, interface_feats], dim=-1)
        relation_features = self.fusion(fused)  # (B, 512)

        return relation_features


if __name__ == "__main__":
    # Shape verification
    from models.backbone import PointNet2Backbone

    backbone = PointNet2Backbone(part_embed_dim=16, max_parts=8)
    relation_module = AssemblyRelationModule(
        backbone_feat_dim=640, num_heads=4, hidden_dim=256, dropout=0.1)

    dummy = torch.randn(2, 2048, 7)
    dummy[:, :, 6] = torch.randint(0, 2, (2, 2048)).float()

    backbone_out = backbone(dummy)
    relation_feat = relation_module(backbone_out, dummy)
    print(f"Relation features: {relation_feat.shape}")
    print(f"Relation module params: {sum(p.numel() for p in relation_module.parameters()):,}")
