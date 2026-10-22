"""CLIP-RN101 TAMamba ReID network used by the four-expert system."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ssm import BidirectionalSelectiveSSM


def _make_bn(dim: int) -> nn.BatchNorm1d:
    layer = nn.BatchNorm1d(dim)
    layer.bias.requires_grad_(False)
    return layer


class MidLevelMultiRangeFusion(nn.Module):
    """Fuse short-, middle-, and Mamba long-range Stage-3 context."""

    def __init__(self, channels: int, num_parts: int, mid_dim: int = 512) -> None:
        super().__init__()
        if num_parts < 1 or mid_dim <= 0:
            raise ValueError("M3F dimensions must be positive")
        self.num_parts = num_parts
        self.short_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.mid_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.long_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.range_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 3, 3, bias=True),
        )
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
        )
        nn.init.zeros_(self.range_gate[-1].weight)
        nn.init.zeros_(self.range_gate[-1].bias)

    def forward(
        self,
        feature_map: Tensor,
        row_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        short = self.short_branch(feature_map)
        middle = self.mid_branch(feature_map)
        long_map = row_states.transpose(1, 2).unsqueeze(-1)
        long_map = long_map.expand(-1, -1, -1, feature_map.shape[-1])
        long_range = self.long_projection(long_map)
        stacked = torch.stack((short, middle, long_range), dim=1)
        weights = self.range_gate(
            torch.cat((short, middle, long_range), dim=1)
        ).softmax(dim=1)
        fused = (stacked * weights[:, :, None, None, None]).sum(dim=1)
        fused = self.output_projection(fused)
        global_feature = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        local_features = F.adaptive_avg_pool2d(
            fused, (self.num_parts, 1)
        ).squeeze(-1).transpose(1, 2)
        return global_feature, local_features, fused


class WaveletMambaDetailResidual(nn.Module):
    """Inject Mamba-gated Haar detail bands into Stage-3."""

    def __init__(self, channels: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("WMDR hidden_dim must be positive")
        self.detail_projection = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.context_gate = nn.Conv2d(channels, hidden_dim, 1, bias=True)
        self.output_projection = nn.Conv2d(
            hidden_dim, channels, 1, bias=False
        )
        nn.init.zeros_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)
        nn.init.zeros_(self.output_projection.weight)

    @staticmethod
    def _haar_details(feature_map: Tensor) -> Tensor:
        height, width = feature_map.shape[-2:]
        if height % 2 or width % 2:
            feature_map = F.pad(
                feature_map,
                (0, width % 2, 0, height % 2),
                mode="replicate",
            )
        top_left = feature_map[:, :, 0::2, 0::2]
        top_right = feature_map[:, :, 0::2, 1::2]
        bottom_left = feature_map[:, :, 1::2, 0::2]
        bottom_right = feature_map[:, :, 1::2, 1::2]
        horizontal = 0.5 * (
            top_left - top_right + bottom_left - bottom_right
        )
        vertical = 0.5 * (
            top_left + top_right - bottom_left - bottom_right
        )
        diagonal = 0.5 * (
            top_left - top_right - bottom_left + bottom_right
        )
        return torch.cat((horizontal, vertical, diagonal), dim=1)

    def forward(
        self, feature_map: Tensor, row_states: Tensor
    ) -> Tensor:
        details = self.detail_projection(self._haar_details(feature_map))
        context = row_states.transpose(1, 2).unsqueeze(-1)
        context = F.interpolate(
            context,
            size=details.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        details = details * (2.0 * torch.sigmoid(self.context_gate(context)))
        residual = F.interpolate(
            self.output_projection(details),
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return feature_map + residual


class DiverseEmbeddingExpansion(nn.Module):
    """Generate training-only identity-centered residual embeddings."""

    def __init__(self, channels: int, num_expansions: int = 2) -> None:
        super().__init__()
        if num_expansions < 1:
            raise ValueError("IC-DEE requires at least one expansion")
        self.num_expansions = num_expansions
        self.expanders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, channels, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels, channels, bias=False),
                )
                for _ in range(num_expansions)
            ]
        )
        for expander in self.expanders:
            nn.init.zeros_(expander[-1].weight)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("IC-DEE input must have shape [batch, channels]")
        return torch.stack(
            [features + expander(features) for expander in self.expanders],
            dim=1,
        )


class CrossModalMaskedPartPrototypeReconstructor(nn.Module):
    """Predict masked Stage-3 body parts from opposite-modality prototypes."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 256,
        num_parts: int = 6,
        mask_ratio: float = 0.5,
        ssm_state: int = 16,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden_dim <= 0:
            raise ValueError("CM-MPPR dimensions must be positive")
        if num_parts < 2:
            raise ValueError("CM-MPPR requires at least two parts")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("CM-MPPR mask ratio must be within (0, 1)")
        self.num_parts = num_parts
        self.mask_ratio = mask_ratio
        self.input_norm = nn.LayerNorm(channels)
        self.input_projection = nn.Linear(channels, hidden_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.empty(1, num_parts, hidden_dim)
        )
        nn.init.normal_(self.position_embedding, std=hidden_dim**-0.5)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_ssm = BidirectionalSelectiveSSM(
            hidden_dim, hidden_dim, ssm_state
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(
            hidden_dim, channels, bias=False
        )

    def forward(
        self,
        feature_map: Tensor,
        labels: Tensor,
        modalities: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        tokens = F.adaptive_avg_pool2d(
            feature_map, (self.num_parts, 1)
        ).squeeze(-1).transpose(1, 2)
        batch = tokens.shape[0]
        targets = torch.empty_like(tokens)
        has_target = torch.zeros(
            batch, dtype=torch.bool, device=tokens.device
        )
        for index in range(batch):
            donors = labels.eq(labels[index]) & modalities.ne(
                modalities[index]
            )
            if donors.any():
                targets[index] = tokens[donors].mean(dim=0).detach()
                has_target[index] = True
            else:
                targets[index] = tokens[index].detach()
        mask = (
            torch.rand(batch, self.num_parts, device=tokens.device)
            < self.mask_ratio
        )
        for index in range(batch):
            if not has_target[index]:
                mask[index] = False
                continue
            if not mask[index].any():
                mask[index, torch.randint(self.num_parts, (), device=tokens.device)] = True
            if mask[index].all():
                mask[index, torch.randint(self.num_parts, (), device=tokens.device)] = False
        encoded = self.input_projection(self.input_norm(tokens))
        masked = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.expand(batch, self.num_parts, -1),
            encoded,
        )
        masked = masked + self.position_embedding
        contextual = masked + self.context_ssm(self.context_norm(masked))
        predictions = self.output_projection(self.output_norm(contextual))
        return predictions, targets, mask


class TAMambaReID(nn.Module):
    """Full TAMamba representation for global and spatial experts."""

    embedding_dim = 8192
    num_parts = 3

    def __init__(
        self, num_classes: int, *, pretrained: str | None = "openai"
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Install open_clip_torch to build the CLIP-RN101 backbone"
            ) from exc

        clip_model = None
        if pretrained == "openai":
            cached_openai = Path.home() / ".cache" / "clip" / "RN101.pt"
            if cached_openai.is_file():
                clip_model = open_clip.load_openai_model(
                    "RN101",
                    device="cpu",
                    cache_dir=str(cached_openai.parent),
                )
        if clip_model is None:
            clip_model = open_clip.create_model(
                "RN101",
                pretrained=pretrained,
                require_pretrained=pretrained is not None,
            )
        self.visual = clip_model.visual
        self.thermal_stem = nn.Sequential(
            copy.deepcopy(self.visual.conv1),
            copy.deepcopy(self.visual.bn1),
            copy.deepcopy(self.visual.act1),
            copy.deepcopy(self.visual.conv2),
            copy.deepcopy(self.visual.bn2),
            copy.deepcopy(self.visual.act2),
            copy.deepcopy(self.visual.conv3),
            copy.deepcopy(self.visual.bn3),
            copy.deepcopy(self.visual.act3),
            copy.deepcopy(self.visual.avgpool),
        )
        self.ssm = BidirectionalSelectiveSSM(1024, 256, 16)
        self.inject_gate = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
        )
        nn.init.zeros_(self.inject_gate[-1].weight)
        nn.init.constant_(self.inject_gate[-1].bias, -2.0)
        self.modality_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1024, 256, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.GELU(),
                    nn.Conv2d(256, 1024, 1, bias=False),
                )
                for _ in range(2)
            ]
        )
        for adapter in self.modality_adapters:
            nn.init.zeros_(adapter[-1].weight)

        self.global_bn = _make_bn(2048)
        self.local_bn = _make_bn(2048)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.local_classifier = nn.Linear(2048, num_classes, bias=False)
        self.mid_fusion = MidLevelMultiRangeFusion(1024, self.num_parts, 512)
        self.mid_global_bn = _make_bn(512)
        self.mid_local_bn = _make_bn(512)
        self.mid_classifier = nn.Linear(512, num_classes, bias=False)
        self.mid_local_classifier = nn.Linear(512, num_classes, bias=False)
        self.dee = DiverseEmbeddingExpansion(2048, 2)
        self.dee_bn = _make_bn(2048)
        self.dee_classifier = nn.Linear(2048, num_classes, bias=False)
        self.wmdr = WaveletMambaDetailResidual(1024, 256)
        self.masked_part_reconstructor = None

    def _visible_stem(self, images: Tensor) -> Tensor:
        visual = self.visual
        for convolution, batch_norm, activation in (
            (visual.conv1, visual.bn1, visual.act1),
            (visual.conv2, visual.bn2, visual.act2),
            (visual.conv3, visual.bn3, visual.act3),
        ):
            images = activation(batch_norm(convolution(images)))
        return visual.avgpool(images)

    def _stem(self, images: Tensor, modalities: Tensor) -> Tensor:
        output: Tensor | None = None
        for modality in (0, 1):
            mask = modalities == modality
            if not mask.any():
                continue
            branch = (
                self._visible_stem(images[mask])
                if modality == 0
                else self.thermal_stem(images[mask])
            )
            if output is None:
                output = branch.new_empty((images.shape[0], *branch.shape[1:]))
            output[mask] = branch
        assert output is not None
        return output

    def _features(
        self, images: Tensor, modalities: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        visual = self.visual
        feature_map = visual.layer2(visual.layer1(self._stem(images, modalities)))
        stage3 = visual.layer3(feature_map)
        rows = stage3.mean(dim=3).transpose(1, 2)
        states = self.ssm(rows)
        gate = torch.sigmoid(self.inject_gate(rows))
        stage3 = stage3 + (gate * states).transpose(1, 2).unsqueeze(-1)
        adapted = stage3.clone()
        for modality, adapter in enumerate(self.modality_adapters):
            mask = modalities == modality
            if mask.any():
                adapted[mask] = stage3[mask] + adapter(stage3[mask])
        stage3 = adapted
        row_states = stage3.mean(dim=3).transpose(1, 2)
        stage3 = self.wmdr(stage3, row_states)
        return visual.layer4(stage3), row_states, stage3

    def forward(
        self,
        images: Tensor,
        modalities: Tensor,
        labels: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        """Return descriptors and training heads; modalities are 0 (RGB) or 1 (IR)."""
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1] != 3:
            raise ValueError("images must have shape [nonempty batch, 3, height, width]")
        if modalities.shape != (images.shape[0],) or modalities.dtype != torch.long:
            raise ValueError("modalities must be a torch.long tensor of shape [batch]")
        if modalities.device != images.device:
            raise ValueError("images and modalities must be on the same device")
        if not ((modalities == 0) | (modalities == 1)).all():
            raise ValueError("modalities must contain only 0 (RGB) and 1 (IR)")
        if labels is not None:
            if labels.shape != modalities.shape or labels.dtype != torch.long:
                raise ValueError("labels must be a torch.long tensor of shape [batch]")
            if labels.device != images.device:
                raise ValueError("images and labels must be on the same device")

        stage4, row_states, stage3 = self._features(images, modalities)
        prediction = target = reconstruction_mask = None
        if self.training and self.masked_part_reconstructor is not None and labels is not None:
            prediction, target, reconstruction_mask = self.masked_part_reconstructor(
                stage3, labels, modalities
            )

        global_raw = F.adaptive_avg_pool2d(stage4, 1).flatten(1)
        local_raw = F.adaptive_avg_pool2d(
            stage4, (self.num_parts, 1)
        ).squeeze(-1).transpose(1, 2)
        global_feat = self.global_bn(global_raw)
        batch, parts, channels = local_raw.shape
        local_feat = self.local_bn(
            local_raw.reshape(batch * parts, channels)
        ).reshape(batch, parts, channels)
        global_logits = self.classifier(global_feat)
        local_logits = self.local_classifier(
            local_feat.reshape(batch * parts, channels)
        ).reshape(batch, parts, -1)

        mid_global_raw, mid_local_raw, _ = self.mid_fusion(stage3, row_states)
        mid_global_feat = self.mid_global_bn(mid_global_raw)
        mid_batch, mid_parts, mid_channels = mid_local_raw.shape
        mid_local_feat = self.mid_local_bn(
            mid_local_raw.reshape(mid_batch * mid_parts, mid_channels)
        ).reshape(mid_batch, mid_parts, mid_channels)
        mid_global_logits = self.mid_classifier(mid_global_feat)
        mid_local_logits = self.mid_local_classifier(
            mid_local_feat.reshape(mid_batch * mid_parts, mid_channels)
        ).reshape(mid_batch, mid_parts, -1)

        dee_raw = self.dee(global_raw)
        expansion_count = dee_raw.shape[1]
        dee_feat = self.dee_bn(
            dee_raw.reshape(batch * expansion_count, channels)
        ).reshape(batch, expansion_count, channels)
        dee_logits = self.dee_classifier(
            dee_feat.reshape(batch * expansion_count, channels)
        ).reshape(batch, expansion_count, -1)

        pieces = [
            F.normalize(local_feat, dim=-1).flatten(1),
            F.normalize(mid_global_feat, dim=1),
            F.normalize(mid_local_feat, dim=-1).flatten(1),
        ]
        embedding = F.normalize(torch.cat(pieces, dim=1), dim=1)
        return {
            "embedding": embedding,
            "global_raw": global_raw,
            "global_feat": global_feat,
            "local_raw": local_raw,
            "local_feat": local_feat,
            "global_logits": global_logits,
            "local_logits": local_logits,
            "mid_global_raw": mid_global_raw,
            "mid_global_feat": mid_global_feat,
            "mid_local_raw": mid_local_raw,
            "mid_local_feat": mid_local_feat,
            "mid_global_logits": mid_global_logits,
            "mid_local_logits": mid_local_logits,
            "dee_raw": dee_raw,
            "dee_source": global_raw,
            "dee_logits": dee_logits,
            "row_states": row_states,
            "spatial_map": stage3,
            "masked_part_prediction": prediction,
            "masked_part_target": target,
            "masked_part_mask": reconstruction_mask,
        }


class CMMPPRTAMambaReID(TAMambaReID):
    """Full TAMamba representation with cross-modal masked-part supervision."""

    def __init__(
        self, num_classes: int, *, pretrained: str | None = "openai"
    ) -> None:
        super().__init__(num_classes, pretrained=pretrained)
        self.masked_part_reconstructor = CrossModalMaskedPartPrototypeReconstructor(
            channels=1024, hidden_dim=256, num_parts=6, mask_ratio=0.5, ssm_state=16
        )
