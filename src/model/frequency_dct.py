"""
frequency_dct.py
----------------
DCT-based Frequency Branch (Novel contribution — detects spectral artifacts).

Architecture (from plan.txt Section 4.4):
  Input : Raw face crops BEFORE CLIP normalization → [B, T, 3, 224, 224]
  Output: y_f ∈ R^{B × 256}

Steps:
  1. Convert to grayscale [B, T, 1, 224, 224]
  2. Split into 8×8 patches → [B, T, 784, 64]  (28×28 = 784 patches)
  3. Apply DCT-II to each 8×8 patch
  4. Extract spectral statistics (mean, std, log-energy) per frame → [B, T, 192]
  5. Temporal pool (mean + std) → [B, 384]
  6. MLP projection → [B, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _dct_matrix(N: int) -> torch.Tensor:
    """
    Build the NxN DCT-II orthonormal matrix.

    D[k, n] = alpha_k * cos(pi*(2n+1)*k / (2*N))
    alpha_0 = 1/sqrt(N),  alpha_k = sqrt(2/N) for k>0
    """
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    D = torch.cos(np.pi * (2 * n + 1) * k / (2 * N))      # [N, N]

    alpha = torch.ones(N) * np.sqrt(2.0 / N)
    alpha[0] = 1.0 / np.sqrt(N)
    D = D * alpha.unsqueeze(1)                              # normalize rows
    return D                                                # [N, N]


class FrequencyDCTBranch(nn.Module):
    """
    Frequency-domain deepfake artifact detector using block-DCT.

    Deepfake generators introduce artifacts in frequency space
    (checkerboard patterns, missing high-frequency detail) that are
    invisible to RGB-domain models. This branch detects those anomalies.

    Args:
        patch_size  : Size of each DCT block (8, matching JPEG standard)
        hidden_dim  : MLP hidden dimension (512)
        output_dim  : Output feature dimension (256)
        dropout     : MLP dropout rate (0.3)
    """

    def __init__(
        self,
        patch_size: int = 8,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim

        # Pre-compute DCT matrix — not learnable
        D = _dct_matrix(patch_size)                 # [8, 8]
        self.register_buffer("D", D)

        # Number of DCT coefficients per patch
        num_dct_coefs = patch_size * patch_size      # 64

        # Per-frame statistics: mean(64) + std(64) + log-energy(64) = 192
        stats_per_frame = num_dct_coefs * 3          # 192

        # Temporal pool: mean + std across T → 2 * 192 = 384
        mlp_input_dim = stats_per_frame * 2          # 384

        # MLP: 384 → 512 → 256
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

        self.last_energy = None                     # Save for visualization

    def _apply_dct(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DCT-II to a batch of 8×8 patches.

        Args:
            patches: [B*T, num_patches, 8, 8]
        Returns:
                    [B*T, num_patches, 8, 8]
        """
        # F = D @ patch @ D^T
        D = self.D                                   # [8, 8]
        F = torch.einsum('ij, ...jk, lk -> ...il', D, patches, D)
        return F

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: [B, T, 3, H, W]  — raw uint8 or float frames (0-255 range)
                   MUST be un-normalized (before CLIP mean/std normalization)
        Returns:
            y_f  : [B, 256]
        """
        B, T, C, H, W = x_raw.shape
        P = self.patch_size

        # ── Step 1: Grayscale ──────────────────────────────────────────────
        # Ensure float [0, 255]
        x = x_raw.float()
        if x.max() <= 1.01:
            x = x * 255.0                           # if already [0,1], scale up

        gray = (0.299 * x[:, :, 0, :, :]
                + 0.587 * x[:, :, 1, :, :]
                + 0.114 * x[:, :, 2, :, :])         # [B, T, H, W]

        # ── Step 2: Extract 8×8 patches using unfold ──────────────────────
        # Reshape to (B*T, 1, H, W) for unfold
        gray_flat = gray.reshape(B * T, 1, H, W)

        # num_patches_per_dim = H // P = 224 // 8 = 28
        # Total patches = 28 * 28 = 784
        patches = gray_flat.unfold(2, P, P).unfold(3, P, P)
        # → [B*T, 1, 28, 28, 8, 8]
        BT, _, nph, npw, _, _ = patches.shape
        num_patches = nph * npw                     # 784
        patches = patches.reshape(BT, num_patches, P, P)   # [B*T, 784, 8, 8]

        # ── Step 3: Apply 2D DCT to each patch ────────────────────────────
        dct_patches = self._apply_dct(patches)      # [B*T, 784, 8, 8]

        # ── Step 4: Spectral statistics per frame ─────────────────────────
        # Mean DCT coefficients across patches: [B*T, 64]
        mu = dct_patches.mean(dim=1).reshape(BT, -1)

        # Std of DCT coefficients across patches: [B*T, 64]
        sigma = dct_patches.std(dim=1).reshape(BT, -1)

        # Log-energy: [B*T, 64]
        energy = torch.log1p((dct_patches ** 2).mean(dim=1)).reshape(BT, -1)
        self.last_energy = energy.detach()

        # Concatenate → [B*T, 192]
        frame_feat = torch.cat([mu, sigma, energy], dim=-1)

        # Reshape to [B, T, 192]
        frame_feat = frame_feat.reshape(B, T, -1)

        # ── Step 5: Temporal pooling ───────────────────────────────────────
        feat_mean = frame_feat.mean(dim=1)          # [B, 192]
        feat_std  = frame_feat.std(dim=1)           # [B, 192]
        freq_feat = torch.cat([feat_mean, feat_std], dim=-1)  # [B, 384]

        # ── Step 6: MLP projection ─────────────────────────────────────────
        y_f = self.mlp(freq_feat)                   # [B, 256]

        return y_f

    def get_dct_energy(self) -> torch.Tensor:
        """
        Returns the [B, T, 8, 8] average log-energy spectrum from the last pass.
        Note: The energy is [B*T, 64]. We reshape to [B, T, 8, 8].
        """
        if self.last_energy is not None:
            # Assuming B=1 during inference, but handling generic B
            BT = self.last_energy.shape[0]
            # Since we don't have T saved easily in the class, we rely on the 
            # caller to know B and T, or we just return the flat [BT, 64] and 
            # let the caller reshape. Let's return [BT, 8, 8]
            return self.last_energy.view(BT, 8, 8)
        return None
