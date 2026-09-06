"""
PointNet++ Backbone Adapted for 7D Assembly Point Clouds
=========================================================
Input:  (B, N, 7)  →  [x, y, z, nx, ny, nz, part_id]

Architecture:
  1. Part ID → Learnable Embedding (part_embed_dim)
  2. Initial features = [normals (3) | part_embedding (E)] → D_in = 3 + E
  3. SA Layer 1 (MSG): N → 512 centroids, multi-scale feature extraction
  4. SA Layer 2 (MSG): 512 → 128 centroids, deeper multi-scale features
  5. Global Set Abstraction: 128 → 1 global feature vector

Output: (B, global_feat_dim) – a single feature vector per assembly sample
        Also returns intermediate features for the Assembly Relationship module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.pointnet2_utils import (
    SetAbstractionMSG,
    GlobalSetAbstraction,
    farthest_point_sample,
    index_points,
)


class PartEmbedding(nn.Module):
    """Learnable embedding for categorical part_id values."""

    def __init__(self, max_parts: int = 8, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(max_parts, embed_dim)

    def forward(self, part_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            part_ids: (B, N) integer part identifiers

        Returns:
            embeddings: (B, N, embed_dim)
        """
        return self.embedding(part_ids.long())


class PointNet2Backbone(nn.Module):
    """
    PointNet++ feature extraction backbone for 7D assembly point clouds.

    Hierarchically extracts local and global geometric features while
    preserving multi-part identity information through learnable embeddings.
    """

    def __init__(self, part_embed_dim: int = 16, max_parts: int = 8):
        super().__init__()

        self.part_embed_dim = part_embed_dim
        self.part_embedding = PartEmbedding(max_parts, part_embed_dim)

        # Initial feature dim: normals (3) + part_embedding (E)
        in_features = 3 + part_embed_dim

        # SA Layer 1 (MSG): N → 512 centroids
        # Input: xyz (3) + features (in_features)
        self.sa1 = SetAbstractionMSG(
            npoint=512,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 64],
            in_channel=in_features,
            mlp_channels_list=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128],
            ]
        )
        # Output features: 64 + 128 + 128 = 320

        # SA Layer 2 (MSG): 512 → 128 centroids
        self.sa2 = SetAbstractionMSG(
            npoint=128,
            radii=[0.2, 0.4, 0.8],
            nsamples=[32, 64, 128],
            in_channel=320,
            mlp_channels_list=[
                [64, 64, 128],
                [128, 128, 256],
                [128, 128, 256],
            ]
        )
        # Output features: 128 + 256 + 256 = 640

        # Global aggregation: 128 → 1 global vector
        self.global_sa = GlobalSetAbstraction(
            in_channel=640 + 3,  # features + xyz
            mlp_channels=[256, 512, 1024]
        )

        self.global_feat_dim = 1024

    def forward(self, point_cloud: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            point_cloud: (B, N, 7) – [x, y, z, nx, ny, nz, part_id]

        Returns:
            dict with:
                'global_feat':  (B, 1024) global feature vector
                'sa1_xyz':      (B, 512, 3)   SA1 centroid coordinates
                'sa1_features': (B, 512, 320)  SA1 features
                'sa2_xyz':      (B, 128, 3)   SA2 centroid coordinates
                'sa2_features': (B, 128, 640)  SA2 features
                'part_ids':     (B, N)        original part IDs
        """
        B, N, _ = point_cloud.shape

        # Decompose input
        xyz = point_cloud[:, :, :3].contiguous()      # (B, N, 3)
        normals = point_cloud[:, :, 3:6].contiguous()  # (B, N, 3)
        part_ids = point_cloud[:, :, 6].contiguous()   # (B, N)

        # Part embedding
        part_embeds = self.part_embedding(part_ids)    # (B, N, E)

        # Initial features: normals + part embedding
        features = torch.cat([normals, part_embeds], dim=-1)  # (B, N, 3+E)

        # Hierarchical feature learning
        sa1_xyz, sa1_features = self.sa1(xyz, features)
        # sa1_xyz: (B, 512, 3), sa1_features: (B, 512, 320)

        sa2_xyz, sa2_features = self.sa2(sa1_xyz, sa1_features)
        # sa2_xyz: (B, 128, 3), sa2_features: (B, 128, 640)

        # Global feature: concatenate xyz with features for global SA
        sa2_combined = torch.cat([sa2_xyz, sa2_features], dim=-1)  # (B, 128, 643)
        global_feat = self.global_sa(sa2_xyz, sa2_combined)
        # global_feat: (B, 1024)

        return {
            "global_feat": global_feat,
            "sa1_xyz": sa1_xyz,
            "sa1_features": sa1_features,
            "sa2_xyz": sa2_xyz,
            "sa2_features": sa2_features,
            "part_ids": part_ids,
        }


if __name__ == "__main__":
    # Shape verification
    model = PointNet2Backbone(part_embed_dim=16, max_parts=8)
    dummy = torch.randn(2, 2048, 7)
    dummy[:, :, 6] = torch.randint(0, 2, (2, 2048)).float()

    out = model(dummy)
    print(f"Global feature: {out['global_feat'].shape}")
    print(f"SA1 xyz: {out['sa1_xyz'].shape}, features: {out['sa1_features'].shape}")
    print(f"SA2 xyz: {out['sa2_xyz'].shape}, features: {out['sa2_features'].shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
