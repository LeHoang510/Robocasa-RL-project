"""
ACT (Action Chunking with Transformers) policy with ResNet18 visual encoder.

Architecture
------------
  ResNet18Encoder   : per-camera image → (n_patches, hidden_dim) spatial tokens
  ACTPolicy (CVAE)  :
    Encoder (train) : [CLS, proprio, action_chunk] → TransformerEncoder → (mu, logvar)
    Decoder         : queries(chunk_size) × context[z, proprio, image_tokens]
                      → TransformerDecoder → action_chunk
  ACTAgent          : wraps ACTPolicy, handles obs buffer + temporal ensemble

Reference: Zhao et al. 2023 "Learning Fine-Grained Bimanual Manipulation
           with Low-Cost Hardware" (ACT paper).
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import Optional


# ---------------------------------------------------------------------------
# 2-D sinusoidal positional encoding for spatial feature maps
# ---------------------------------------------------------------------------

def _build_2d_pos_enc(h: int, w: int, dim: int, device=None) -> torch.Tensor:
    """Returns (h*w, dim) positional encoding."""
    assert dim % 4 == 0
    d4 = dim // 4
    y_pos = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1) / h
    x_pos = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(1) / w
    div   = torch.exp(torch.arange(0, d4, dtype=torch.float32, device=device) *
                      (-math.log(10000.0) / d4))
    y_enc = torch.cat([torch.sin(y_pos * div), torch.cos(y_pos * div)], dim=1)  # (h, d4*2)
    x_enc = torch.cat([torch.sin(x_pos * div), torch.cos(x_pos * div)], dim=1)  # (w, d4*2)
    # broadcast to (h, w, dim) then flatten
    enc = torch.cat([
        y_enc.unsqueeze(1).expand(h, w, d4 * 2),
        x_enc.unsqueeze(0).expand(h, w, d4 * 2),
    ], dim=2)                                          # (h, w, dim)
    return enc.reshape(h * w, dim)


# ---------------------------------------------------------------------------
# ResNet18 visual encoder
# ---------------------------------------------------------------------------

class ResNet18Encoder(nn.Module):
    """
    ResNet18 backbone (pretrained on ImageNet).
    Removes avgpool + fc, projects spatial feature map to hidden_dim.

    Input : (B, 3, H, W)  — normalised float, e.g. 128×128
    Output: (B, n_patches, hidden_dim)  where n_patches = (H/32) * (W/32)
    """

    def __init__(self, hidden_dim: int = 512, img_h: int = 128, img_w: int = 128,
                 pretrained: bool = True):
        super().__init__()
        backbone  = tv_models.resnet18(weights="DEFAULT" if pretrained else None)
        # Keep everything up to (but not including) avgpool
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Determine actual feature map size via dummy forward pass —
        # img_h // 32 is wrong for non-power-of-2 sizes (e.g. 84 → 3×3, not 2×2)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_h, img_w)
            dummy = self.stem(dummy)
            dummy = self.layer1(dummy); dummy = self.layer2(dummy)
            dummy = self.layer3(dummy); dummy = self.layer4(dummy)
            feat_h, feat_w = dummy.shape[2], dummy.shape[3]

        self.n_patches = feat_h * feat_w

        self.proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.register_buffer(
            "pos_enc",
            _build_2d_pos_enc(feat_h, feat_w, hidden_dim),     # (n_patches, hidden_dim)
        )

        # ImageNet normalisation (applied to [0,1] float images)
        self.register_buffer("img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.img_mean) / self.img_std
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)           # (B, 512, h, w)
        x = self.proj(x)                                  # (B, hidden_dim, h, w)
        B, C, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)   # (B, n_patches, hidden_dim)
        x = x + self.pos_enc.unsqueeze(0)                 # add 2-D pos enc
        return x


# ---------------------------------------------------------------------------
# ACT Policy (CVAE + Transformer)
# ---------------------------------------------------------------------------

class ACTPolicy(nn.Module):
    """
    ACT: CVAE encoder (training only) + Transformer decoder.

    Parameters
    ----------
    obs_dim      : proprioceptive obs dimension (e.g. 25)
    act_dim      : action dimension (e.g. 12)
    chunk_size   : number of actions predicted at once (e.g. 100)
    hidden_dim   : transformer hidden dimension (default 512)
    n_heads      : number of attention heads (default 8)
    enc_layers   : CVAE encoder transformer layers (default 4)
    dec_layers   : decoder transformer layers (default 7)
    latent_dim   : CVAE latent dimension (default 32)
    n_cameras    : number of camera views (default 2)
    img_h/img_w  : image resolution (default 128)
    kl_weight    : β for β-VAE KL loss (default 10.0)
    dropout      : transformer dropout (default 0.1)
    """

    def __init__(
        self,
        obs_dim:    int   = 25,
        act_dim:    int   = 12,
        chunk_size: int   = 40,
        hidden_dim: int   = 512,
        n_heads:    int   = 8,
        enc_layers: int   = 4,
        dec_layers: int   = 7,
        latent_dim: int   = 32,
        n_cameras:  int   = 2,
        img_h:      int   = 128,
        img_w:      int   = 128,
        kl_weight:  float = 1e-3,
        dropout:    float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.act_dim    = act_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight  = kl_weight
        self.n_cameras  = n_cameras

        # -- Visual backbones (one per camera, shared weights) ----------------
        self.backbone = ResNet18Encoder(
            hidden_dim=hidden_dim, img_h=img_h, img_w=img_w,
            pretrained=pretrained_backbone,
        )
        n_img_tokens = self.backbone.n_patches * n_cameras   # e.g. 16*2=32

        # -- Projections ------------------------------------------------------
        self.proprio_proj  = nn.Linear(obs_dim,    hidden_dim)
        self.action_proj   = nn.Linear(act_dim,    hidden_dim)
        self.latent_proj   = nn.Linear(latent_dim, hidden_dim)

        # -- CVAE Encoder (used only during training) -------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=enc_layers, enable_nested_tensor=False)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # 1-D positional enc for encoder: [CLS, proprio, a_0, ..., a_{T-1}]
        self.enc_pos_enc   = nn.Embedding(1 + 1 + chunk_size, hidden_dim)
        self.mu_proj       = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj   = nn.Linear(hidden_dim, latent_dim)

        # -- Decoder ----------------------------------------------------------
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder     = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)   # learned queries
        self.action_head = nn.Linear(hidden_dim, act_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    # CVAE Encoder
    # ------------------------------------------------------------------

    def _encode(self, proprio: torch.Tensor, actions: torch.Tensor):
        """
        proprio : (B, obs_dim)
        actions : (B, chunk_size, act_dim)
        Returns  : mu (B, latent_dim), logvar (B, latent_dim)
        """
        B = proprio.shape[0]
        cls    = self.cls_token.expand(B, -1, -1)              # (B, 1, H)
        p_tok  = self.proprio_proj(proprio).unsqueeze(1)        # (B, 1, H)
        a_tok  = self.action_proj(actions)                      # (B, T, H)

        seq = torch.cat([cls, p_tok, a_tok], dim=1)            # (B, 1+1+T, H)
        pos = self.enc_pos_enc(
            torch.arange(seq.shape[1], device=seq.device)
        ).unsqueeze(0)
        seq = seq + pos
        enc = self.cvae_encoder(seq)                           # (B, 1+1+T, H)

        cls_out = enc[:, 0]                                    # (B, H)
        return self.mu_proj(cls_out), self.logvar_proj(cls_out)

    def _reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # ------------------------------------------------------------------
    # Visual context
    # ------------------------------------------------------------------

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        images : (B, n_cameras, 3, H, W)  float32 in [0, 1]
        Returns: (B, n_cameras * n_patches, hidden_dim)
        """
        B, C, _, H, W = images.shape
        imgs = images.reshape(B * C, 3, H, W)
        feats = self.backbone(imgs)                            # (B*C, n_p, H)
        return feats.reshape(B, C * self.backbone.n_patches, -1)

    def _build_context(self, images: torch.Tensor, proprio: torch.Tensor,
                       z: torch.Tensor) -> torch.Tensor:
        """Concatenate [z_token, proprio_token, image_tokens] as decoder memory."""
        z_tok = self.latent_proj(z).unsqueeze(1)               # (B, 1, H)
        p_tok = self.proprio_proj(proprio).unsqueeze(1)        # (B, 1, H)
        i_tok = self._encode_images(images)                    # (B, n_img, H)
        return torch.cat([z_tok, p_tok, i_tok], dim=1)         # (B, 2+n_img, H)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images:  torch.Tensor,
        proprio: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ):
        """
        Training  (actions provided) → returns (loss, recon_loss, kl_loss)
        Inference (actions=None)     → returns (chunk_size, act_dim) action chunk
        """
        B = images.shape[0]
        device = images.device

        if actions is not None:
            # --- Training ---
            mu, logvar = self._encode(proprio, actions)
            z          = self._reparameterise(mu, logvar)
        else:
            # --- Inference: use prior mean z=0 ---
            mu = logvar = z = torch.zeros(B, self.latent_dim, device=device)

        context = self._build_context(images, proprio, z)      # (B, N, H)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, T, H)
        dec_out = self.decoder(tgt=queries, memory=context)    # (B, T, H)
        pred    = self.action_head(dec_out)                    # (B, T, act_dim)

        if actions is None:
            return pred                                        # (B, chunk_size, act_dim)

        # Losses
        recon_loss = F.l1_loss(pred, actions)
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.kl_weight * kl_loss
        return loss, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "config": self._cfg()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ACTPolicy":
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device)

    def _cfg(self) -> dict:
        bb = self.backbone
        return dict(
            obs_dim    = self.proprio_proj.in_features,
            act_dim    = self.act_dim,
            chunk_size = self.chunk_size,
            hidden_dim = bb.proj.out_channels,
            n_heads    = self.cvae_encoder.layers[0].self_attn.num_heads,
            enc_layers = len(self.cvae_encoder.layers),
            dec_layers = len(self.decoder.layers),
            latent_dim = self.latent_dim,
            n_cameras  = self.n_cameras,
            img_h      = int(bb.stem[0].weight.shape[2]) * 2,
            img_w      = int(bb.stem[0].weight.shape[3]) * 2,
            kl_weight  = self.kl_weight,
        )


# ---------------------------------------------------------------------------
# ACT inference agent with temporal ensemble
# ---------------------------------------------------------------------------

class ACTAgent:
    """
    Wraps ACTPolicy for step-by-step inference.

    At each step:
      1. Predict a new action chunk (chunk_size actions).
      2. Add it to a buffer of overlapping predictions.
      3. Return the temporally-ensembled action (exp-weighted average).

    temporal_agg_gamma : higher → prefer most recent chunk.
    """

    def __init__(self, policy: ACTPolicy, device: str = "cpu",
                 temporal_agg_gamma: float = 0.01):
        self.policy  = policy.to(device).eval()
        self.device  = device
        self.gamma   = temporal_agg_gamma
        self.chunk_size = policy.chunk_size
        self._buffer: list[tuple[int, np.ndarray]] = []  # (predicted_at_step, chunk)
        self._step   = 0

    def reset(self):
        self._buffer = []
        self._step   = 0

    @torch.no_grad()
    def predict(self, images: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        """
        images : (n_cameras, H, W, 3)  uint8  0-255
        proprio: (obs_dim,)             float32
        Returns: (act_dim,)             float32  — single action to execute
        """
        # Predict a new chunk at every step (re-plan each time)
        img_t = (torch.from_numpy(images).float() / 255.0)     # (C, H, W, 3)
        img_t = img_t.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)  # (1, C, 3, H, W)
        pro_t = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)

        chunk = self.policy(img_t, pro_t)                       # (1, T, act_dim)
        chunk_np = chunk.squeeze(0).cpu().numpy()               # (T, act_dim)
        self._buffer.append((self._step, chunk_np))

        # Temporal ensemble: weight by recency
        weights, actions = [], []
        for (t0, ck) in self._buffer:
            offset = self._step - t0
            if offset >= self.chunk_size:
                continue
            w = math.exp(-self.gamma * offset)
            weights.append(w)
            actions.append(ck[offset])

        self._buffer = [(t0, ck) for (t0, ck) in self._buffer
                        if self._step - t0 < self.chunk_size]

        total_w = sum(weights)
        action = sum(w * a for w, a in zip(weights, actions)) / total_w

        self._step += 1
        return action.astype(np.float32)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kw) -> "ACTAgent":
        policy = ACTPolicy.load(path, device=device)
        return cls(policy, device=device, **kw)
