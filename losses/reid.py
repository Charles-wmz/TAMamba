"""Fixed composite supervision for the Full TAMamba four-expert system."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CrossBatchMemory(nn.Module):
    """Detached global feature queue for cross-modal hard mining."""

    def __init__(
        self,
        capacity: int = 2048,
        min_entries: int = 256,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if capacity <= 0:
            raise ValueError("XBM capacity must be positive")
        if not 0 <= min_entries <= capacity:
            raise ValueError("XBM min_entries must be within [0, capacity]")
        if temperature <= 0.0:
            raise ValueError("XBM temperature must be positive")
        self.capacity = capacity
        self.min_entries = min_entries
        self.temperature = temperature
        self.register_buffer("feature_queue", torch.empty(0, 0))
        self.register_buffer(
            "label_queue", torch.empty(0, dtype=torch.long)
        )
        self.register_buffer(
            "modality_queue", torch.empty(0, dtype=torch.long)
        )

    @property
    def size(self) -> int:
        return int(self.label_queue.numel())

    def forward(
        self,
        features: Tensor,
        labels: Tensor,
        modalities: Tensor,
    ) -> Tensor:
        normalized = F.normalize(features.float(), dim=1)
        loss = normalized.sum() * 0.0
        if self.size >= self.min_entries and self.size > 0:
            current_similarity = normalized @ normalized.t()
            positive = labels[:, None].eq(
                labels[None, :]
            ) & modalities[:, None].ne(modalities[None, :])
            memory_similarity = normalized @ self.feature_queue.t()
            negative = labels[:, None].ne(
                self.label_queue[None, :]
            ) & modalities[:, None].ne(self.modality_queue[None, :])
            valid = positive.any(dim=1) & negative.any(dim=1)
            if valid.any():
                hardest_positive = current_similarity.masked_fill(
                    ~positive, float("inf")
                ).min(dim=1).values
                hardest_negative = memory_similarity.masked_fill(
                    ~negative, float("-inf")
                ).max(dim=1).values
                ranking = (
                    hardest_negative[valid] - hardest_positive[valid]
                ) / self.temperature
                loss = self.temperature * F.softplus(ranking).mean()
        self._enqueue(
            normalized.detach(),
            labels.detach(),
            modalities.detach(),
        )
        return loss

    @torch.no_grad()
    def _enqueue(
        self, features: Tensor, labels: Tensor, modalities: Tensor
    ) -> None:
        if self.size:
            features = torch.cat((self.feature_queue, features), dim=0)
            labels = torch.cat((self.label_queue, labels), dim=0)
            modalities = torch.cat(
                (self.modality_queue, modalities), dim=0
            )
        self.feature_queue = features[-self.capacity :]
        self.label_queue = labels[-self.capacity :]
        self.modality_queue = modalities[-self.capacity :]


def batch_hard_triplet_loss(
    features: Tensor, labels: Tensor, margin: float = 0.3
) -> Tensor:
    normalized = F.normalize(features, dim=1)
    distances = torch.cdist(normalized.float(), normalized.float(), p=2)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = labels[:, None].ne(labels[None, :])
    valid = same.any(dim=1) & different.any(dim=1)
    if not valid.any():
        return normalized.sum() * 0.0
    hardest_positive = distances.masked_fill(
        ~same, float("-inf")
    ).max(dim=1).values
    hardest_negative = distances.masked_fill(
        ~different, float("inf")
    ).min(dim=1).values
    return F.relu(
        hardest_positive[valid] - hardest_negative[valid] + margin
    ).mean()


def _heterogeneous_centers(
    features: Tensor, labels: Tensor, modalities: Tensor
) -> tuple[Tensor, Tensor, Tensor] | None:
    centers: list[Tensor] = []
    center_labels: list[Tensor] = []
    center_modalities: list[Tensor] = []
    for pid in labels.unique():
        for modality in (0, 1):
            mask = labels.eq(pid) & modalities.eq(modality)
            if mask.any():
                centers.append(features[mask].mean(dim=0))
                center_labels.append(pid)
                center_modalities.append(labels.new_tensor(modality))
    if len(centers) < 4:
        return None
    return (
        F.normalize(torch.stack(centers), dim=1),
        torch.stack(center_labels),
        torch.stack(center_modalities),
    )


def heterogeneous_center_triplet_loss(
    features: Tensor,
    labels: Tensor,
    modalities: Tensor,
    margin: float,
) -> Tensor:
    centers = _heterogeneous_centers(features, labels, modalities)
    if centers is None:
        return features.sum() * 0.0
    stacked, center_labels, center_modalities = centers
    distances = torch.cdist(stacked.float(), stacked.float(), p=2)
    cross_modal = center_modalities[:, None].ne(
        center_modalities[None, :]
    )
    positive = center_labels[:, None].eq(
        center_labels[None, :]
    ) & cross_modal
    negative = center_labels[:, None].ne(
        center_labels[None, :]
    ) & cross_modal
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0
    positive_distance = distances.masked_fill(
        ~positive, float("-inf")
    ).max(dim=1).values
    negative_distance = distances.masked_fill(
        ~negative, float("inf")
    ).min(dim=1).values
    return F.relu(
        positive_distance[valid] - negative_distance[valid] + margin
    ).mean()


def soft_margin_heterogeneous_center_loss(
    features: Tensor, labels: Tensor, modalities: Tensor
) -> Tensor:
    """Non-saturating HCT used by Local-HCT and IC-DEE centers."""
    centers = _heterogeneous_centers(features, labels, modalities)
    if centers is None:
        return features.sum() * 0.0
    stacked, center_labels, center_modalities = centers
    distances = torch.cdist(stacked.float(), stacked.float(), p=2)
    cross_modal = center_modalities[:, None].ne(
        center_modalities[None, :]
    )
    positive = center_labels[:, None].eq(
        center_labels[None, :]
    ) & cross_modal
    negative = center_labels[:, None].ne(
        center_labels[None, :]
    ) & cross_modal
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0
    positive_distance = distances.masked_fill(
        ~positive, float("-inf")
    ).max(dim=1).values
    negative_distance = distances.masked_fill(
        ~negative, float("inf")
    ).min(dim=1).values
    return F.softplus(
        positive_distance[valid] - negative_distance[valid]
    ).mean()


def paired_cross_modal_loss(
    features: Tensor, labels: Tensor, modalities: Tensor
) -> Tensor:
    normalized = F.normalize(features, dim=1)
    losses: list[Tensor] = []
    for pid in labels.unique():
        visible = normalized[labels.eq(pid) & modalities.eq(0)]
        thermal = normalized[labels.eq(pid) & modalities.eq(1)]
        pair_count = min(visible.shape[0], thermal.shape[0])
        if pair_count:
            losses.append(
                1.0
                - F.cosine_similarity(
                    visible[:pair_count],
                    thermal[:pair_count],
                    dim=1,
                ).mean()
            )
    return torch.stack(losses).mean() if losses else features.sum() * 0.0


def cross_modal_supcon_loss(
    features: Tensor,
    labels: Tensor,
    modalities: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    if temperature <= 0.0:
        raise ValueError("SupCon temperature must be positive")
    normalized = F.normalize(features, dim=1)
    logits = normalized @ normalized.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    cross_modal = modalities[:, None].ne(modalities[None, :])
    positives = labels[:, None].eq(labels[None, :]) & cross_modal
    log_prob = logits - torch.logsumexp(
        logits.masked_fill(~cross_modal, float("-inf")),
        dim=1,
        keepdim=True,
    )
    valid = positives.any(dim=1)
    if not valid.any():
        return normalized.sum() * 0.0
    positive_log_prob = (
        log_prob.masked_fill(~positives, 0.0).sum(dim=1)
        / positives.sum(dim=1).clamp_min(1)
    )
    return -positive_log_prob[valid].mean()


def row_state_alignment_loss(
    row_states: Tensor,
    labels: Tensor,
    modalities: Tensor,
) -> Tensor:
    losses: list[Tensor] = []
    for pid in labels.unique():
        visible = row_states[labels.eq(pid) & modalities.eq(0)]
        thermal = row_states[labels.eq(pid) & modalities.eq(1)]
        if visible.numel() and thermal.numel():
            losses.append(
                1.0
                - F.cosine_similarity(
                    visible.mean(dim=0),
                    thermal.mean(dim=0),
                    dim=-1,
                ).mean()
            )
    return torch.stack(losses).mean() if losses else row_states.sum() * 0.0


def paired_spatial_alignment_loss(
    feature_map: Tensor,
    labels: Tensor,
    modalities: Tensor,
) -> Tensor:
    normalized = F.normalize(feature_map, dim=1)
    losses: list[Tensor] = []
    for pid in labels.unique():
        visible = normalized[labels.eq(pid) & modalities.eq(0)]
        thermal = normalized[labels.eq(pid) & modalities.eq(1)]
        pair_count = min(visible.shape[0], thermal.shape[0])
        if pair_count:
            losses.append(
                1.0
                - (
                    visible[:pair_count] * thermal[:pair_count]
                ).sum(dim=1).mean()
            )
    return (
        torch.stack(losses).mean()
        if losses
        else feature_map.sum() * 0.0
    )


def diverse_embedding_diversity_loss(
    original: Tensor, expansions: Tensor
) -> Tensor:
    if expansions.ndim != 3:
        raise ValueError("IC-DEE expansions must have shape [B, K, C]")
    original_normalized = F.normalize(original, dim=1)
    expansion_normalized = F.normalize(expansions, dim=-1)
    diversity = (
        expansion_normalized * original_normalized.unsqueeze(1)
    ).sum(dim=-1).mean()
    count = expansion_normalized.shape[1]
    if count > 1:
        pairwise = expansion_normalized @ expansion_normalized.transpose(
            1, 2
        )
        off_diagonal = ~torch.eye(
            count,
            dtype=torch.bool,
            device=expansions.device,
        ).unsqueeze(0)
        diversity = diversity + pairwise.masked_select(
            off_diagonal
        ).mean()
    return diversity


def identity_center_guided_dee_loss(
    expansions: Tensor,
    labels: Tensor,
    modalities: Tensor,
) -> Tensor:
    if expansions.ndim != 3:
        raise ValueError("IC-DEE expansions must have shape [B, K, C]")
    return torch.stack(
        [
            soft_margin_heterogeneous_center_loss(
                expansions[:, index], labels, modalities
            )
            for index in range(expansions.shape[1])
        ]
    ).mean()


def cross_modal_masked_part_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "CM-MPPR prediction and target must share shape [B, P, C]"
        )
    if mask.shape != prediction.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("CM-MPPR mask must be boolean with shape [B, P]")
    if not mask.any():
        zero = prediction.sum() * 0.0
        return zero, zero.detach(), zero.detach()
    predicted = F.normalize(prediction[mask].float(), dim=1)
    expected = F.normalize(target.detach()[mask].float(), dim=1)
    cosine = (predicted * expected).sum(dim=1)
    return (
        (1.0 - cosine).mean(),
        cosine.detach().mean(),
        mask.float().detach().mean(),
    )


def _identity_losses(
    outputs: dict[str, Tensor], labels: Tensor
) -> tuple[Tensor, Tensor]:
    global_loss = sum(
        F.cross_entropy(outputs[key], labels, label_smoothing=0.1)
        for key in ("global_logits", "mid_global_logits")
    )
    local_loss = sum(
        torch.stack(
            [
                F.cross_entropy(
                    outputs[key][:, part], labels, label_smoothing=0.1
                )
                for part in range(outputs[key].shape[1])
            ]
        ).sum()
        for key in ("local_logits", "mid_local_logits")
    )
    return global_loss, local_loss


def reid_loss(
    outputs: dict[str, Tensor],
    labels: Tensor,
    modalities: Tensor,
    *,
    xbm: CrossBatchMemory,
    spatial: bool,
    epoch: int,
) -> tuple[Tensor, dict[str, float]]:
    """Apply the fixed Full objective for one of the four trained experts."""
    embedding = outputs["embedding"]
    global_raw = outputs["global_raw"]
    id_global, id_local = _identity_losses(outputs, labels)
    triplet = batch_hard_triplet_loss(embedding, labels, margin=0.3)
    hct = heterogeneous_center_triplet_loss(
        global_raw, labels, modalities, margin=0.3
    )
    local_hct = torch.stack(
        [
            soft_margin_heterogeneous_center_loss(
                outputs[key][:, part], labels, modalities
            )
            for key in ("local_raw", "mid_local_raw")
            for part in range(outputs[key].shape[1])
        ]
    ).mean()
    xbm_loss = xbm(global_raw, labels, modalities)
    pair = paired_cross_modal_loss(global_raw, labels, modalities)
    supcon = cross_modal_supcon_loss(global_raw, labels, modalities)
    alignment = row_state_alignment_loss(
        outputs["row_states"], labels, modalities
    )
    spatial_pair = paired_spatial_alignment_loss(
        outputs["spatial_map"], labels, modalities
    )
    dee_raw = outputs["dee_raw"]
    dee_logits = outputs["dee_logits"]
    dee_id = torch.stack(
        [
            F.cross_entropy(
                dee_logits[:, index], labels, label_smoothing=0.1
            )
            for index in range(dee_logits.shape[1])
        ]
    ).mean()
    dee_diversity = diverse_embedding_diversity_loss(
        outputs["dee_source"], dee_raw
    )
    dee_center = identity_center_guided_dee_loss(dee_raw, labels, modalities)

    reconstruction = global_raw.sum() * 0.0
    reconstruction_cosine = global_raw.new_zeros(())
    masked_fraction = global_raw.new_zeros(())
    reconstruction_weight = 0.0
    if outputs.get("masked_part_prediction") is not None:
        reconstruction, reconstruction_cosine, masked_fraction = (
            cross_modal_masked_part_reconstruction_loss(
                outputs["masked_part_prediction"],
                outputs["masked_part_target"],
                outputs["masked_part_mask"],
            )
        )
        reconstruction_weight = 0.2 * min(epoch / 10.0, 1.0)
    total = (
        0.5 * id_global
        + id_local
        + triplet
        + 0.5 * hct
        + 0.2 * local_hct
        + 0.2 * xbm_loss
        + pair
        + 0.5 * supcon
        + 0.05 * alignment
        + (0.1 if spatial else 0.0) * spatial_pair
        + 0.5 * dee_id
        + 0.2 * dee_diversity
        + 0.5 * dee_center
        + reconstruction_weight * reconstruction
    )
    values = {
        "loss": float(total.detach()),
        "id": float((0.5 * id_global + id_local).detach()),
        "triplet": float(triplet.detach()),
        "hct": float(hct.detach()),
        "local_hct": float(local_hct.detach()),
        "xbm": float(xbm_loss.detach()),
        "pair": float(pair.detach()),
        "supcon": float(supcon.detach()),
        "alignment": float(alignment.detach()),
        "spatial_pair": float(spatial_pair.detach()),
        "dee_id": float(dee_id.detach()),
        "dee_diversity": float(dee_diversity.detach()),
        "dee_center": float(dee_center.detach()),
        "masked_part_reconstruction": float(reconstruction.detach()),
        "masked_part_reconstruction_cosine": float(
            reconstruction_cosine.detach()
        ),
        "masked_part_fraction": float(masked_fraction.detach()),
    }
    return total, values
