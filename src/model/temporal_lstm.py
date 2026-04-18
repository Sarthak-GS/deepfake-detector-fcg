"""
temporal_lstm.py
----------------
Bi-LSTM Temporal Branch (Novel contribution — replaces DFD-FCG's Conv2D temporal affinity).

Architecture (from plan.txt Section 4.3):
  Input : Per-frame CLS embeddings from last 4 ViT layers → [B, T, 1024]
  Output: y_t ∈ R^{B × 512}

Steps:
  1. Average CLS tokens from ViT layers 21-24
  2. Compute frame-to-frame temporal differences
  3. Concatenate original + difference → [B, T, 2048]
  4. Linear projection → [B, T, 512]
  5. 2-layer Bidirectional LSTM → [B, T, 512]
  6. Attention pooling over time → [B, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBiLSTM(nn.Module):
    """
    Bidirectional LSTM temporal branch for deepfake detection.

    Takes per-frame CLS token embeddings from the ViT encoder and models
    temporal anomalies (flickering, identity drift, unnatural transitions).

    Args:
        input_dim  : Dimensionality of each CLS token (1024 for ViT-L/14)
        hidden_dim : LSTM hidden size per direction (256 → 512 bidirectional)
        num_layers : Number of stacked LSTM layers (2)
        dropout    : Dropout between LSTM layers (0.3)
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project [CLS ‖ Δ(CLS)] from 2*input_dim → 512
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.GELU(),
        )

        # 2-layer Bidirectional LSTM
        # input_size=512, hidden=hidden_dim, bidirectional → output=hidden_dim*2
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,         # 256 per direction
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2       # 512 (bidirectional)

        # Soft attention pooling: score each time-step
        self.attn_fc = nn.Linear(lstm_out_dim, 1)

        self.output_dim = lstm_out_dim      # 512 for downstream classifier

        self.last_attn_weights = None       # To store attention for visualization

    def forward(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_tokens: [B, T, D]  — averaged CLS tokens across selected layers
        Returns:
            y_t       : [B, 512]   — temporally aggregated representation
        """
        B, T, D = cls_tokens.shape

        # ── Step 2: Temporal difference features ──────────────────────────
        # diff[t] = cls[t] - cls[t-1], zero-padded at t=0
        diff = cls_tokens[:, 1:, :] - cls_tokens[:, :-1, :]     # [B, T-1, D]
        zero_pad = torch.zeros(B, 1, D, device=cls_tokens.device,
                               dtype=cls_tokens.dtype)
        diff = torch.cat([zero_pad, diff], dim=1)                # [B, T, D]

        # ── Step 3: Concatenate ────────────────────────────────────────────
        lstm_in = torch.cat([cls_tokens, diff], dim=-1)          # [B, T, 2D]

        # ── Step 4: Linear projection ──────────────────────────────────────
        lstm_in = self.input_proj(lstm_in)                       # [B, T, 512]

        # ── Step 5: Bi-LSTM ────────────────────────────────────────────────
        lstm_out, _ = self.lstm(lstm_in)                         # [B, T, 512]

        # ── Step 6: Attention pooling ──────────────────────────────────────
        attn_logits = self.attn_fc(lstm_out)                     # [B, T, 1]
        attn_weights = F.softmax(attn_logits, dim=1)             # [B, T, 1]
        self.last_attn_weights = attn_weights.detach()           # Save for vis
        y_t = (lstm_out * attn_weights).sum(dim=1)               # [B, 512]

        return y_t
        
    def get_attention_weights(self) -> torch.Tensor:
        """Returns the [B, T] attention weights from the last forward pass."""
        if self.last_attn_weights is not None:
            return self.last_attn_weights.squeeze(-1)
        return None
