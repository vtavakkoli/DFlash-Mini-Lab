from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class QwenDFlashConfig:
    model_id: str
    target_hidden_size: int
    target_vocab_size: int
    candidate_size: int
    target_layer_ids: tuple[int, ...] = (4, 14, 27)
    memory_tokens: int = 16
    block_size: int = 6
    draft_dim: int = 256
    draft_heads: int = 8
    draft_layers: int = 2
    loss_gamma: float = 3.0

    @property
    def raw_memory_dim(self) -> int:
        return self.target_hidden_size * len(self.target_layer_ids)

    @property
    def speculative_tokens(self) -> int:
        return self.block_size - 1


class HiddenFusionDrafter(nn.Module):
    """Parallel DFlash-style drafter over fused verifier hidden-state memory."""

    def __init__(self, config: QwenDFlashConfig, candidate_head_weight: torch.Tensor):
        super().__init__()
        self.config = config
        self.fusion = nn.Linear(config.raw_memory_dim, config.draft_dim)
        self.fusion_norm = nn.LayerNorm(config.draft_dim)
        self.anchor_proj = nn.Linear(config.target_hidden_size, config.draft_dim)
        self.slot_emb = nn.Embedding(config.block_size, config.draft_dim)
        self.mask_slots = nn.Parameter(torch.zeros(config.speculative_tokens, config.draft_dim))
        nn.init.normal_(self.mask_slots, mean=0.0, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=config.draft_dim,
            nhead=config.draft_heads,
            dim_feedforward=config.draft_dim * 3,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.draft_layers)
        self.final_norm = nn.LayerNorm(config.draft_dim)
        self.output_proj = nn.Linear(config.draft_dim, config.target_hidden_size, bias=False)
        self.candidate_bias = nn.Parameter(torch.zeros(config.candidate_size))
        self.register_buffer(
            "candidate_head_weight",
            candidate_head_weight.detach().float().contiguous(),
            persistent=True,
        )

    def forward(
        self,
        raw_memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        anchor_embedding: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.fusion_norm(self.fusion(raw_memory.float()))
        batch = raw_memory.shape[0]
        slot_ids = torch.arange(self.config.block_size, device=raw_memory.device)
        slot = self.slot_emb(slot_ids).unsqueeze(0).expand(batch, -1, -1)
        anchor = self.anchor_proj(anchor_embedding.float()).unsqueeze(1)
        masks = self.mask_slots.unsqueeze(0).expand(batch, -1, -1)
        target = torch.cat([anchor, masks], dim=1) + slot
        hidden = self.final_norm(
            self.decoder(
                tgt=target,
                memory=memory,
                memory_key_padding_mask=memory_padding_mask.bool(),
            )
        )
        future = self.output_proj(hidden[:, 1:, :])
        return torch.einsum("bph,ch->bpc", future, self.candidate_head_weight) + self.candidate_bias


@dataclass
class TrainingTensors:
    raw_memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    anchor_embedding: torch.Tensor
    targets_local: torch.Tensor

    def __len__(self) -> int:
        return int(self.targets_local.shape[0])


def build_training_tensors(
    trajectories: list[dict],
    candidate_ids: list[int],
    embedding_weight: torch.Tensor,
    config: QwenDFlashConfig,
) -> TrainingTensors:
    local = {int(token): i for i, token in enumerate(candidate_ids)}
    memories: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    targets: list[list[int]] = []
    emb = embedding_weight.detach().float().cpu()

    for item in trajectories:
        token_ids = [int(x) for x in item["token_ids"]]
        prompt_len = int(item["prompt_len"])
        raw_hidden = item["raw_hidden"].float().cpu()
        if int(raw_hidden.shape[0]) != len(token_ids):
            raise ValueError("trajectory hidden-state length does not match token ids")

        # Anchors are verifier-generated tokens. Target hidden memory ends just
        # before the known anchor because the anchor is not yet in the KV cache.
        first_anchor = prompt_len
        last_anchor_exclusive = len(token_ids) - config.speculative_tokens
        for anchor_idx in range(first_anchor, last_anchor_exclusive):
            future = token_ids[anchor_idx + 1 : anchor_idx + config.block_size]
            if len(future) != config.speculative_tokens:
                continue
            if any(token not in local for token in future):
                continue
            prefix_memory = raw_hidden[:anchor_idx]
            keep = prefix_memory[-config.memory_tokens :]
            pad_count = config.memory_tokens - int(keep.shape[0])
            if pad_count:
                pad = torch.zeros(pad_count, config.raw_memory_dim, dtype=torch.float32)
                keep = torch.cat([pad, keep], dim=0)
            mask = torch.zeros(config.memory_tokens, dtype=torch.bool)
            if pad_count:
                mask[:pad_count] = True
            memories.append(keep)
            padding_masks.append(mask)
            anchors.append(emb[token_ids[anchor_idx]])
            targets.append([local[token] for token in future])

    if not memories:
        raise RuntimeError("no Qwen DFlash training examples were constructed")
    return TrainingTensors(
        raw_memory=torch.stack(memories).float(),
        memory_padding_mask=torch.stack(padding_masks),
        anchor_embedding=torch.stack(anchors).float(),
        targets_local=torch.tensor(targets, dtype=torch.long),
    )


def train_drafter(
    config: QwenDFlashConfig,
    candidate_head_weight: torch.Tensor,
    tensors: TrainingTensors,
    *,
    steps: int = 180,
    batch_size: int = 16,
    learning_rate: float = 1.5e-3,
    seed: int = 7,
) -> tuple[HiddenFusionDrafter, dict]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = HiddenFusionDrafter(config, candidate_head_weight)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    positions = torch.arange(config.speculative_tokens, dtype=torch.float32)
    static_weights = torch.exp(-positions / max(float(config.loss_gamma), 1e-6))
    ema_accuracy = torch.full((config.speculative_tokens,), 0.5, dtype=torch.float32)
    last_loss = 0.0

    for _ in range(max(1, int(steps))):
        idx = torch.randint(0, len(tensors), (min(int(batch_size), len(tensors)),), generator=generator)
        logits = model(
            tensors.raw_memory[idx],
            tensors.memory_padding_mask[idx],
            tensors.anchor_embedding[idx],
        )
        target = tensors.targets_local[idx]
        per_token = F.cross_entropy(
            logits.reshape(-1, config.candidate_size),
            target.reshape(-1),
            reduction="none",
        ).view(target.shape[0], config.speculative_tokens)
        with torch.no_grad():
            batch_accuracy = (logits.argmax(dim=-1) == target).float().mean(dim=0)
            ema_accuracy.mul_(0.9).add_(0.1 * batch_accuracy)
            adaptive = 1.0 + 0.75 * (1.0 - ema_accuracy)
            weights = static_weights * adaptive
            weights = weights / weights.mean()
        loss = (per_token * weights.unsqueeze(0)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())

    model.eval()
    correct = torch.zeros(config.speculative_tokens, dtype=torch.float64)
    count = 0
    with torch.inference_mode():
        for start in range(0, len(tensors), 32):
            stop = min(start + 32, len(tensors))
            logits = model(
                tensors.raw_memory[start:stop],
                tensors.memory_padding_mask[start:stop],
                tensors.anchor_embedding[start:stop],
            )
            target = tensors.targets_local[start:stop]
            correct += (logits.argmax(dim=-1) == target).double().sum(dim=0)
            count += int(target.shape[0])

    metrics = {
        "examples": len(tensors),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "final_loss": last_loss,
        "position_accuracy": [float(x / max(count, 1)) for x in correct.tolist()],
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_aux_parameter_count": sum(p.numel() for p in model.parameters()),
        "loss_objective": "prefix-decay + detached EMA bottleneck weighting (D-PACE-inspired, not exact D-PACE)",
    }
    return model, metrics


def save_bundle(
    path: str | Path,
    config: QwenDFlashConfig,
    candidate_ids: list[int],
    model: HiddenFusionDrafter,
    metadata: dict,
) -> None:
    payload = {
        "format_version": 1,
        "config": {**asdict(config), "target_layer_ids": list(config.target_layer_ids)},
        "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_bundle(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = dict(payload["config"])
    raw["target_layer_ids"] = tuple(int(x) for x in raw["target_layer_ids"])
    config = QwenDFlashConfig(**raw)
    state = payload["state_dict"]
    candidate_head_weight = state["candidate_head_weight"].float()
    model = HiddenFusionDrafter(config, candidate_head_weight)
    model.load_state_dict(state)
    model.eval()
    return config, payload["candidate_ids"].long(), model, payload.get("metadata", {})
