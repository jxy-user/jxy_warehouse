import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(text_ids)  # [B, T, D]
        pooled = self.pool(emb.transpose(1, 2)).squeeze(-1)
        return pooled


class StructEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MMCANet(nn.Module):
    """Minimal multimodal cross-attention style fusion model."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(cfg["image_in_channels"], cfg["image_embed_dim"])
        self.text_encoder = TextEncoder(cfg["text_vocab_size"], cfg["text_embed_dim"])
        self.struct_encoder = StructEncoder(cfg["num_struct_features"], cfg["struct_hidden_dim"], cfg["fusion_dim"])

        self.img_to_fusion = nn.Linear(cfg["image_embed_dim"], cfg["fusion_dim"])
        self.txt_to_fusion = nn.Linear(cfg["text_embed_dim"], cfg["fusion_dim"])

        self.attn = nn.MultiheadAttention(embed_dim=cfg["fusion_dim"], num_heads=4, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(cfg["fusion_dim"] * 3, cfg["fusion_dim"]), nn.Sigmoid())
        self.classifier = nn.Linear(cfg["fusion_dim"], cfg["num_classes"])

    def forward(self, image: torch.Tensor, text_ids: torch.Tensor, struct_features: torch.Tensor) -> torch.Tensor:
        img = self.img_to_fusion(self.image_encoder(image))
        txt = self.txt_to_fusion(self.text_encoder(text_ids))
        tab = self.struct_encoder(struct_features)

        q = img.unsqueeze(1)
        kv = torch.stack([txt, tab], dim=1)
        attn_out, _ = self.attn(q, kv, kv)
        attn_out = attn_out.squeeze(1)

        merged = torch.cat([img, attn_out, tab], dim=-1)
        gate = self.gate(merged)
        fused = gate * img + (1 - gate) * attn_out
        return self.classifier(fused)
