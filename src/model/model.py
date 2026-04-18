"""
model.py
--------
EnhancedDeepfakeDetector: Full 3-branch multimodal deepfake detector.

Architecture (plan.txt Sections 4, 5):
  - Branch A (Spatial)  : FCG-style synoptic attention on ViT per-layer keys/embeddings
  - Branch B (Temporal) : Bi-LSTM on CLS tokens from last 4 ViT layers (NEW)
  - Branch C (Frequency): DCT spectral analysis on raw face crops (NEW)

Backbone: FROZEN OpenCLIP ViT-L/14 (preloaded weights, no gradient updates)
Trainable: Synoptic Decoder + Bi-LSTM + DCT Branch + 4 classifier heads (~15M params)
"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import open_clip
from typing import List, Optional, Tuple, Dict

from .temporal_lstm import TemporalBiLSTM
from .frequency_dct import FrequencyDCTBranch


# ─────────────────────────────────────────────
# 1. Focal Loss (plan.txt Section 5.7)
# ─────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 4.0,
    weight: Optional[List[float]] = None,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """
    Focal loss for binary classification with label smoothing.
      FL(p, y) = -alpha_y * (1 - p_y)^gamma * log(p_y)

    Args:
        logits  : [N, 2]  — raw logits
        targets : [N]     — int labels (0=real, 1=fake)
        gamma   : focusing parameter (4.0 per plan.txt)
        weight  : per-class weights [w_real, w_fake]
        label_smoothing : smoothing factor (0.05)
    Returns:
        scalar mean loss
    """
    num_classes = logits.shape[-1]
    log_p = F.log_softmax(logits, dim=-1)
    log_p = torch.clamp(log_p, min=-5.0)

    # Apply label smoothing: hard targets → soft targets
    targets_one_hot = F.one_hot(targets.long(), num_classes).float()
    targets_smooth = targets_one_hot * (1.0 - label_smoothing) + label_smoothing / num_classes

    # Focal weight based on hard target probability
    targets_ = targets.long().unsqueeze(1)              # [N, 1]
    log_pt   = log_p.gather(1, targets_).squeeze(-1)   # [N]
    pt       = log_pt.exp()

    if weight is not None:
        w = torch.tensor(weight, device=logits.device, dtype=logits.dtype)
        at = w.gather(0, targets.long())
    else:
        at = 1.0

    focal_weight = at * (1 - pt) ** gamma               # [N]

    # Cross-entropy with smooth labels
    loss = -focal_weight.unsqueeze(-1) * targets_smooth * log_p  # [N, C]
    return loss.sum(dim=-1).mean()


# ─────────────────────────────────────────────
# 2. FCG Guidance Loss (plan.txt Section 5.7b)
# ─────────────────────────────────────────────

def fcg_loss(
    synoptic_queries_with_idx: list,
    face_features: Dict,
    temperature: float = 30.0,
) -> torch.Tensor:
    """
    Aligns learned synoptic queries to pre-extracted face part features.

    For each layer l, synoptic_queries[l] ∈ [B, K, D] should match
    face_features['L{l+1}'] ∈ [K, D] (lips, skin, eyes, nose).

    Args:
        synoptic_queries_with_idx : list of (layer_idx, [B, K, D]) tuples
        face_features    : dict with keys 'L1'..'L24', each [K, D]
        temperature      : cosine similarity scaling (30.0)
    Returns:
        scalar FCG classification loss
    """
    # Get device from first query
    first_sq = synoptic_queries_with_idx[0]
    if isinstance(first_sq, tuple):
        device = first_sq[1].device
    else:
        device = first_sq.device

    total_loss = 0.0
    count = 0

    for item in synoptic_queries_with_idx:
        if isinstance(item, tuple):
            layer_idx, sq = item
        else:
            # Backward compat: assume sequential indexing
            layer_idx = synoptic_queries_with_idx.index(item)
            sq = item

        # Map to 1-indexed layer key 'L1'..'L24'
        layer_key = f"L{layer_idx + 1}"
        if layer_key not in face_features:
            continue

        target_labels = torch.arange(sq.shape[1], device=sq.device)

        # Check if 'k' is nested or direct
        f_data = face_features[layer_key]
        if isinstance(f_data, dict) and 'k' in f_data:
            f = f_data['k']
        else:
            f = f_data

        if not isinstance(f, torch.Tensor):
            f = torch.tensor(f, dtype=sq.dtype, device=sq.device)
        else:
            f = f.to(device=sq.device, dtype=sq.dtype)

        # Normalize
        sq_norm = F.normalize(sq, dim=-1)       # [B, K, D]
        f_norm  = F.normalize(f,  dim=-1)       # [K, D]

        # Similarity: [B, K, K]
        sim = temperature * torch.einsum('bkd,jd->bkj', sq_norm, f_norm)

        B, K, _ = sim.shape
        lbl = target_labels.unsqueeze(0).expand(B, K)  # [B, K]

        # CE loss: for each batch item, each query should match its face part
        loss_l = F.cross_entropy(
            sim.reshape(B * K, K),
            lbl.reshape(B * K)
        )
        total_loss += loss_l
        count += 1

    return total_loss / max(count, 1)


# ─────────────────────────────────────────────
# 3. Synoptic Block — Spatial (FCG) Branch
# ─────────────────────────────────────────────

class SynoBlock(nn.Module):
    """
    Single layer of the FCG Synoptic Decoder (plan.txt Section 4.2).

    Learns K=4 synoptic embeddings (one per face part) and attends to
    the ViT key/embedding outputs via cosine-similarity attention.

    Args:
        embed_dim  : ViT embedding dimension (1024)
        num_synos  : Number of face parts / synoptic queries (4)
        ksize      : Kernel size for conv compression of keys (5)
    """

    def __init__(self, embed_dim: int = 1024, num_synos: int = 4, ksize: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_synos = num_synos

        # Learnable synoptic embeddings — initialized from face features later
        self.synoptic_embedding = nn.Parameter(
            torch.randn(num_synos, embed_dim)
        )

        # Stored attention maps for explainability
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        k_l: torch.Tensor,     # [B, T, 257, 16, 64] — layer keys
        emb_l: torch.Tensor,   # [B, T, 257, D]       — layer embeddings
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            y_s_l : [B, D]         — spatial feature for this layer
            sq    : [B, num_synos, D] — synoptic queries (for FCG loss)
        """
        B, T, _, H, d = k_l.shape
        D = emb_l.shape[-1]

        # Drop CLS token → patch tokens only
        k_patches   = k_l[:, :, 1:, :, :]                  # [B, T, 256, H, d]
        emb_patches = emb_l[:, :, 1:, :]                    # [B, T, 256, D]

        # Flatten multi-head keys → [B, T, 256, D]
        k_flat = k_patches.reshape(B, T, 256, H * d)       # [B, T, 256, 1024]

        # Expand synoptic queries to batch: [B, num_synos, D]
        sq = self.synoptic_embedding.unsqueeze(0).expand(B, -1, -1)

        # Cosine-similarity attention
        sq_norm  = F.normalize(sq,     dim=-1)              # [B, K, D]
        k_norm   = F.normalize(k_flat, dim=-1)              # [B, T, 256, D]

        # [B, T, K, 256]  via einsum
        affinity = 100.0 * torch.einsum('bkd, btpd -> btkp', sq_norm, k_norm)
        alpha    = F.softmax(affinity, dim=-1)              # [B, T, K, 256]

        # Store for explainability
        self._attn_weights = alpha.detach()

        # Attend: [B, T, K, D]
        s_mix = torch.einsum('btkp, btpd -> btkd', alpha, emb_patches)

        # Pool over T and K → [B, D]
        y_s_l = s_mix.mean(dim=(1, 2))

        return y_s_l, sq


# ─────────────────────────────────────────────
# 4. Full Model
# ─────────────────────────────────────────────

class EnhancedDeepfakeDetector(nn.Module):
    """
    Full multimodal deepfake detector (plan.txt Sections 1, 4, 5).

    Combines:
      - FROZEN ViT-L/14 backbone (OpenCLIP)
      - FCG Synoptic Spatial Decoder (24 SynoBlocks)
      - Bi-LSTM Temporal Branch (on last 4 layer CLS tokens)
      - DCT Frequency Branch (on raw un-normalized frames)
      - 4 classification heads + late fusion

    Args:
        face_feature_path : path to misc/L14_real_semantic_patches_v4_2000.pickle
        num_frames        : frames per clip (8)
        temporal_cls_layers : which ViT layers to extract CLS from ([21,22,23,24])
    """

    # ViT-L/14 specs from plan.txt Section 4.1
    VIT_EMBED_DIM  = 1024
    VIT_NUM_LAYERS = 24
    VIT_NUM_HEADS  = 16
    VIT_HEAD_DIM   = 64
    VIT_NUM_PATCHES = 256
    NUM_SYNOS       = 4       # face parts: lips, skin, eyes, nose

    def __init__(
        self,
        face_feature_path: str,
        num_frames: int = 8,
        temporal_cls_layers: List[int] = None,
        # Spatial
        ksize_s: int = 5,
        # Temporal
        temporal_hidden_dim: int = 256,
        temporal_num_layers: int = 2,
        temporal_dropout: float = 0.3,
        # Frequency
        dct_patch_size: int = 8,
        freq_hidden_dim: int = 512,
        freq_output_dim: int = 256,
        freq_dropout: float = 0.3,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.temporal_cls_layers = temporal_cls_layers or [20, 21, 22, 23]

        # Pre-compute which ViT layers to keep attrs for (avoids recomputing in loop)
        self._syno_layer_indices = list(range(0, self.VIT_NUM_LAYERS, 2))  # [0,2,4,...,22]
        self._active_layer_set = set(
            self.temporal_cls_layers + self._syno_layer_indices
        )

        # ── A. FROZEN ViT-L/14 backbone ───────────────────────────────────
        # Load OpenCLIP model; we only use the visual encoder
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k'
        )
        self.visual = clip_model.visual
        self.visual.requires_grad_(False)   # FREEZE all encoder params
        self.visual.eval()

        # We need per-layer attribute access (q, k, v, emb).
        # Handled by _encode_with_attrs() which manually extracts them.

        # ── B. Spatial — FCG Synoptic Decoder ─────────────────────────────
        self.syno_blocks = nn.ModuleList([
            SynoBlock(
                embed_dim=self.VIT_EMBED_DIM,
                num_synos=self.NUM_SYNOS,
                ksize=ksize_s,
            )
            for _ in range(self.VIT_NUM_LAYERS)
        ])

        # Load face features ONCE and reuse for both init and FCG loss
        self.face_features = self._load_face_features(face_feature_path)

        # Initialize synoptic embeddings from face features
        self._init_synoptic_embeddings_from_data(self.face_features)

        # ── C. Temporal — Bi-LSTM ─────────────────────────────────────────
        self.temporal_branch = TemporalBiLSTM(
            input_dim=self.VIT_EMBED_DIM,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
        )
        temporal_out_dim = temporal_hidden_dim * 2          # 512

        # ── D. Frequency — DCT ────────────────────────────────────────────
        self.frequency_branch = FrequencyDCTBranch(
            patch_size=dct_patch_size,
            hidden_dim=freq_hidden_dim,
            output_dim=freq_output_dim,
            dropout=freq_dropout,
        )
        freq_out_dim = freq_output_dim                      # 256

        # ── E. Classification heads ────────────────────────────────────────
        # 3 branch heads + 1 aggregate head
        self.head_spatial = nn.Sequential(
            nn.LayerNorm(self.VIT_EMBED_DIM),
            nn.Linear(self.VIT_EMBED_DIM, 2),
        )
        self.head_temporal = nn.Sequential(
            nn.LayerNorm(temporal_out_dim),
            nn.Linear(temporal_out_dim, 2),
        )
        self.head_frequency = nn.Sequential(
            nn.LayerNorm(freq_out_dim),
            nn.Linear(freq_out_dim, 2),
        )
        agg_dim = self.VIT_EMBED_DIM + temporal_out_dim + freq_out_dim  # 1792
        self.head_aggregate = nn.Sequential(
            nn.LayerNorm(agg_dim),
            nn.Linear(agg_dim, 2),
        )

        # NOTE: self.face_features was already loaded above (line ~311)
        # and used for synoptic embedding init. No need to load again.

    # ──────────────────────────────────────────────────────────────────────
    # Face feature initialization
    # ──────────────────────────────────────────────────────────────────────

    # Face part order for synoptic queries (K=4)
    FACE_PARTS = ['lips', 'skin', 'eyes', 'nose']

    def _load_face_features(self, path: str) -> Optional[Dict]:
        """
        Load pre-extracted face part features for FCG loss.

        The pickle has structure:  data['k'] -> {'eyes': [24 tensors of [1024]], ...}
        We restructure it into:   {'L1': [4, 1024], 'L2': [4, 1024], ...}
        so each L{i} contains stacked features for [lips, skin, eyes, nose].
        """
        try:
            with open(path, 'rb') as f:
                raw = pickle.load(f)
        except (FileNotFoundError, Exception) as e:
            print(f"[WARNING] Could not load face features from {path}: {e}")
            print("[WARNING] FCG loss will be disabled (using cls_loss only).")
            return None

        # Check if already in L1..L24 format
        if 'L1' in raw:
            print(f"[INFO] Face features already in L1..L24 format.")
            return raw

        # Restructure from raw['k']['eyes'][layer] format
        if 'k' not in raw or not isinstance(raw['k'], dict):
            print(f"[WARNING] Unexpected face feature format, keys: {list(raw.keys())}")
            return None

        k_data = raw['k']
        num_layers = len(k_data.get(self.FACE_PARTS[0], []))
        print(f"[INFO] Restructuring face features: {list(k_data.keys())} × {num_layers} layers")

        structured = {}
        for layer_idx in range(num_layers):
            parts = []
            for part_name in self.FACE_PARTS:
                if part_name in k_data and layer_idx < len(k_data[part_name]):
                    feat = k_data[part_name][layer_idx]
                    if not isinstance(feat, torch.Tensor):
                        feat = torch.tensor(feat, dtype=torch.float32)
                    parts.append(feat)
            if parts:
                # Stack [lips, skin, eyes, nose] -> [4, 1024]
                structured[f"L{layer_idx + 1}"] = torch.stack(parts)

        print(f"[INFO] Face features restructured: {len(structured)} layers, "
              f"each [{len(self.FACE_PARTS)}, {parts[0].shape[-1]}]")
        return structured

    def _init_synoptic_embeddings_from_data(self, data: Optional[Dict]):
        """Initialize SynoBlock embeddings from pre-loaded face features dict."""
        if data is None:
            return

        initialized = 0
        try:
            for l, block in enumerate(self.syno_blocks):
                layer_key = f"L{l+1}"
                if layer_key in data:
                    feat = data[layer_key]  # [K=4, D=1024]
                    if isinstance(feat, torch.Tensor):
                        block.synoptic_embedding.data = feat.float()
                    else:
                        block.synoptic_embedding.data = torch.tensor(
                            feat, dtype=torch.float32
                        )
                    initialized += 1
            print(f"[INFO] Initialized {initialized}/{len(self.syno_blocks)} "
                  f"synoptic embeddings from face features.")
        except Exception as e:
            print(f"[WARNING] Could not init synoptic embeddings: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # ViT Encoder with per-layer attribute extraction
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_with_attrs(self, x: torch.Tensor) -> List[Dict]:
        """
        Run the frozen ViT-L/14 and capture per-layer q, k, v, emb.

        Args:
            x: [B, T, 3, 224, 224]  — CLIP-normalized frames
        Returns:
            list of 24 dicts, each with keys: 'q','k','v','emb'
              q, k, v : [B, T, 257, 16, 64]
              emb     : [B, T, 257, 1024]
        """
        B, T, C, H, W = x.shape

        # Flatten B and T for patch embedding
        x_flat = x.reshape(B * T, C, H, W)

        vit = self.visual

        # ViT-L/14 internal forward (mirrors OpenCLIP VisionTransformer)
        # Step 1: Patch projection
        x_p = vit.conv1(x_flat.to(vit.conv1.weight.dtype))     # [BT, D, 16, 16]
        x_p = x_p.flatten(2).transpose(1, 2)                    # [BT, 256, D]

        # Step 2: CLS + positional embedding
        cls = vit.class_embedding.unsqueeze(0).expand(B * T, 1, -1)
        x_p = torch.cat([cls, x_p], dim=1)                      # [BT, 257, D]
        x_p = x_p + vit.positional_embedding
        x_p = vit.ln_pre(x_p)

        # Transpose for transformer: [257, BT, D]
        x_p = x_p.permute(1, 0, 2)

        layer_attrs = []
        D = x_p.shape[-1]   # 1024

        # Step 3: Through each transformer block, collect q, k, v, emb
        for i, block in enumerate(vit.transformer.resblocks):
            # LayerNorm before attention
            normed = block.ln_1(x_p)  # [257, BT, D]

            # Compute q, k, v manually from in_proj_weight
            in_proj_w = block.attn.in_proj_weight   # [3D, D]
            in_proj_b = block.attn.in_proj_bias      # [3D]

            qkv = F.linear(normed, in_proj_w, in_proj_b)  # [257, BT, 3D]
            q, k, v = qkv.chunk(3, dim=-1)                # each [257, BT, D]

            # Reshape to [BT, 257, H, d] then [B, T, 257, H, d]
            H_heads = 16   # ViT-L/14
            d_head  = D // H_heads

            def reshape_qkv(t):
                # t: [257, BT, D]
                t = t.permute(1, 0, 2)                 # [BT, 257, D]
                t = t.view(B, T, 257, H_heads, d_head)
                return t

            q_r = reshape_qkv(q)
            k_r = reshape_qkv(k)
            v_r = reshape_qkv(v)

            # Run the actual block forward
            x_p = block(x_p)     # [257, BT, D]

            # Embedding after block
            emb = x_p.permute(1, 0, 2).view(B, T, 257, D)  # [B, T, 257, D]

            # MEMORY OPTIMIZATION: Only store attrs for layers we actually use
            if i in self._active_layer_set:
                layer_attrs.append({
                    'q':   q_r,    # [B, T, 257, 16, 64]
                    'k':   k_r,    # [B, T, 257, 16, 64]
                    'v':   v_r,    # [B, T, 257, 16, 64]
                    'emb': emb,    # [B, T, 257, 1024]
                })
            else:
                layer_attrs.append(None) # placeholders to keep indices consistent

        return layer_attrs

    # ──────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,         # [B, T, 3, 224, 224] CLIP-normalized
        x_raw: torch.Tensor,     # [B, T, 3, 224, 224] raw (for DCT branch)
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x    : CLIP-normalized frames for ViT + Spatial branch
            x_raw: Un-normalized raw frames for Frequency branch

        Returns:
            (logits_final, logits_s, logits_t, logits_f, logits_a)
            logits_final: [B, 2]  — ensemble prediction
            others      : [B, 2]  — per-branch predictions
        """
        B = x.shape[0]

        # ── A. Encoder: get per-layer attributes (frozen, no grad) ─────────
        layer_attrs = self._encode_with_attrs(x)

        # ── B. Spatial branch (FCG Synoptic Decoder) ──────────────────────
        y_s = torch.zeros(B, self.VIT_EMBED_DIM, device=x.device)
        synoptic_queries = []  # list of (layer_idx, tensor) tuples

        for l, block in enumerate(self.syno_blocks):
            # MEMORY OPTIMIZATION: Only use every 2nd layer for synoptic spatial analysis
            if l % 2 != 0:
                continue

            attrs = layer_attrs[l]
            if attrs is None: continue

            k_l   = attrs['k']     # [B, T, 257, 16, 64]
            emb_l = attrs['emb']   # [B, T, 257, 1024]

            if self.training:
                # Use gradient checkpointing to save VRAM
                y_s_l, sq_l = checkpoint(block, k_l, emb_l, use_reentrant=False)
            else:
                y_s_l, sq_l = block(k_l, emb_l)
                
            y_s = y_s + y_s_l
            synoptic_queries.append((l, sq_l))  # pass actual layer index

        # y_s: [B, 1024]

        # ── C. Temporal branch (Bi-LSTM) ───────────────────────────────────
        # Extract CLS tokens from last 4 layers, average them
        cls_list = []
        for l_idx in self.temporal_cls_layers:
            cls_list.append(layer_attrs[l_idx]['emb'][:, :, 0, :])  # [B, T, D]
        cls_mean = torch.stack(cls_list, dim=0).mean(dim=0)  # [B, T, 1024]
        y_t = self.temporal_branch(cls_mean)                  # [B, 512]

        # ── D. Frequency branch (DCT) ──────────────────────────────────────
        y_f = self.frequency_branch(x_raw)                    # [B, 256]

        # ── E. Classification heads ─────────────────────────────────────────
        logits_s = self.head_spatial(y_s)                     # [B, 2]
        logits_t = self.head_temporal(y_t)                    # [B, 2]
        logits_f = self.head_frequency(y_f)                   # [B, 2]

        y_cat    = torch.cat([y_s, y_t, y_f], dim=-1)        # [B, 1792]
        logits_a = self.head_aggregate(y_cat)                 # [B, 2]

        # ── F. Log-average ensemble (plan.txt Section 5.6) ─────────────────
        p_s = F.softmax(logits_s, dim=-1)
        p_t = F.softmax(logits_t, dim=-1)
        p_f = F.softmax(logits_f, dim=-1)
        p_a = F.softmax(logits_a, dim=-1)
        logits_final = torch.log((p_s + p_t + p_f + p_a) / 4.0 + 1e-4)

        return logits_final, logits_s, logits_t, logits_f, logits_a, synoptic_queries

    # ──────────────────────────────────────────────────────────────────────
    # Loss computation
    # ──────────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        logits_s: torch.Tensor,
        logits_t: torch.Tensor,
        logits_f: torch.Tensor,
        logits_a: torch.Tensor,
        synoptic_queries: List[Optional[torch.Tensor]],
        labels: torch.Tensor,
        cls_weight: float = 10.0,
        ffg_weight: float = 1.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss = cls_loss + ffg_weight * fcg_loss

        Args:
            logits_s/t/f/a : [B, 2] per-branch logits
            synoptic_queries: list of [B, K, D] (some can be None)
            labels          : [B] ground-truth labels
            cls_weight      : scale for focal loss (10.0)
            ffg_weight      : scale for FCG alignment loss (1.5)

        Returns:
            (total_loss, cls_loss, guidance_loss)
        """
        # Focal loss across all 4 heads (concatenated)
        all_logits = torch.cat([logits_s, logits_t, logits_f, logits_a], dim=0)
        all_labels = labels.repeat(4)

        fl = focal_loss(all_logits, all_labels, gamma=4.0, weight=[1.0, 1.0])
        cls_l = fl * cls_weight

        # FCG guidance loss (if face features loaded)
        if self.face_features is not None:
            # synoptic_queries is already a list of (layer_idx, tensor) tuples
            fcg_l = fcg_loss(synoptic_queries, self.face_features, temperature=30.0)
        else:
            fcg_l = torch.tensor(0.0, device=cls_l.device)

        total = cls_l + ffg_weight * fcg_l
        return total, cls_l, fcg_l

    # ──────────────────────────────────────────────────────────────────────
    # Attention maps for explainability
    # ──────────────────────────────────────────────────────────────────────

    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """
        Returns spatial attention maps from all SynoBlocks.
        Shape: [num_layers, B, T, K, 256]  (can reshape 256→16×16)
        """
        maps = []
        for block in self.syno_blocks:
            if block._attn_weights is not None:
                maps.append(block._attn_weights)
        if not maps:
            return None
        return torch.stack(maps, dim=0)   # [L, B, T, K, 256]

    def get_temporal_attention(self) -> Optional[torch.Tensor]:
        """
        Returns the Bi-LSTM attention weights from the last forward pass.
        Shape: [B, T]
        """
        return self.temporal_branch.get_attention_weights()

    def get_dct_spectrum(self) -> Optional[torch.Tensor]:
        """
        Returns the average DCT log-energy spectrum from the last pass.
        Shape: [B*T, 8, 8]
        """
        return self.frequency_branch.get_dct_energy()
