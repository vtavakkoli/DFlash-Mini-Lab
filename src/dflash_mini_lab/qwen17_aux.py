from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Qwen17Config:
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
    selector_rank: int = 32
    jump_hidden: int = 128
    jump_offsets: tuple[int, ...] = (2, 4)
    loss_gamma: float = 3.0

    @property
    def raw_memory_dim(self) -> int:
        return self.target_hidden_size * len(self.target_layer_ids)

    @property
    def speculative_tokens(self) -> int:
        return self.block_size - 1


class HiddenFusionDrafter(nn.Module):
    """DFlash-style parallel block drafter conditioned on frozen target hidden memory."""

    def __init__(self, config: Qwen17Config, candidate_head_weight: torch.Tensor):
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

    def hidden_and_logits(
        self,
        raw_memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        anchor_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        )[:, 1:, :]
        future = self.output_proj(hidden)
        logits = torch.einsum("bph,ch->bpc", future, self.candidate_head_weight) + self.candidate_bias
        return hidden, future, logits

    def forward(
        self,
        raw_memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        anchor_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.hidden_and_logits(raw_memory, memory_padding_mask, anchor_embedding)[2]


class LowRankSelector(nn.Module):
    """Context-gated low-rank predecessor selector shared by DFlash2/MOBS variants."""

    def __init__(self, config: Qwen17Config):
        super().__init__()
        rank = config.selector_rank
        self.context = nn.Linear(config.raw_memory_dim, rank)
        self.anchor = nn.Linear(config.target_hidden_size, rank)
        self.prev = nn.Embedding(config.candidate_size, rank)
        self.next = nn.Embedding(config.candidate_size, rank)
        self.scale = rank ** -0.5

    def gate(self, context_summary: torch.Tensor) -> torch.Tensor:
        return 0.5 + torch.sigmoid(self.context(context_summary.float()))

    def first_logits(self, context_summary: torch.Tensor, anchor_embedding: torch.Tensor) -> torch.Tensor:
        state = self.anchor(anchor_embedding.float()) * self.gate(context_summary)
        return torch.einsum("br,cr->bc", state, self.next.weight) * self.scale

    def transition_logits(self, context_summary: torch.Tensor, prev_local: torch.Tensor) -> torch.Tensor:
        state = self.prev(prev_local.long()) * self.gate(context_summary)
        return torch.einsum("...r,cr->...c", state, self.next.weight) * self.scale


class JumpHead(nn.Module):
    """Separate sparse future-anchor predictor used only by DFlash4."""

    def __init__(self, config: Qwen17Config):
        super().__init__()
        h = config.jump_hidden
        self.context = nn.Linear(config.raw_memory_dim, h)
        self.anchor = nn.Linear(config.target_hidden_size, h)
        self.offset = nn.Embedding(len(config.jump_offsets), h)
        self.norm = nn.LayerNorm(h)
        self.out = nn.Linear(h, config.target_hidden_size, bias=False)

    def forward(self, context_summary: torch.Tensor, anchor_embedding: torch.Tensor) -> torch.Tensor:
        base = self.context(context_summary.float()) + self.anchor(anchor_embedding.float())
        x = torch.tanh(base[:, None, :] + self.offset.weight[None, :, :])
        return self.out(self.norm(x))


class FusedJumpHead(nn.Module):
    """Small residual head over existing drafter slot hidden states for DFlash5."""

    def __init__(self, config: Qwen17Config):
        super().__init__()
        self.positions = tuple(max(0, int(offset) - 1) for offset in config.jump_offsets)
        self.residual = nn.Sequential(
            nn.Linear(config.draft_dim, config.draft_dim),
            nn.Tanh(),
            nn.Linear(config.draft_dim, config.target_hidden_size, bias=False),
        )
        self.offset_scale = nn.Parameter(torch.ones(len(self.positions), config.target_hidden_size))

    def forward(self, draft_hidden: torch.Tensor, base_future: torch.Tensor) -> torch.Tensor:
        index = torch.tensor(self.positions, dtype=torch.long, device=draft_hidden.device)
        latent = draft_hidden.index_select(1, index)
        base = base_future.index_select(1, index)
        residual = self.residual(latent) * self.offset_scale.unsqueeze(0)
        return base + residual


@dataclass
class TrainingTensors:
    raw_memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    context_summary: torch.Tensor
    anchor_embedding: torch.Tensor
    anchor_local: torch.Tensor
    targets_local: torch.Tensor

    def __len__(self) -> int:
        return int(self.targets_local.shape[0])


def _summary(keep: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = ~mask
    if bool(valid.any()):
        return keep[valid].float().mean(dim=0)
    return torch.zeros(keep.shape[-1], dtype=torch.float32)


def build_training_tensors(
    trajectories: list[dict],
    candidate_ids: list[int],
    candidate_embedding_weight: torch.Tensor,
    config: Qwen17Config,
) -> TrainingTensors:
    local = {int(token): i for i, token in enumerate(candidate_ids)}
    memories: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []
    summaries: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    anchor_local: list[int] = []
    targets: list[list[int]] = []
    emb = candidate_embedding_weight.detach().float().cpu()

    for item in trajectories:
        token_ids = [int(x) for x in item["token_ids"]]
        prompt_len = int(item["prompt_len"])
        raw_hidden = item["raw_hidden"].cpu()
        if int(raw_hidden.shape[0]) != len(token_ids):
            raise ValueError("trajectory hidden-state length does not match token ids")
        first_anchor = prompt_len
        last_anchor_exclusive = len(token_ids) - config.speculative_tokens
        for anchor_idx in range(first_anchor, last_anchor_exclusive):
            anchor_token = token_ids[anchor_idx]
            future = token_ids[anchor_idx + 1 : anchor_idx + config.block_size]
            if anchor_token not in local or any(token not in local for token in future):
                continue
            prefix_memory = raw_hidden[:anchor_idx]
            keep = prefix_memory[-config.memory_tokens :]
            pad_count = config.memory_tokens - int(keep.shape[0])
            if pad_count:
                pad = torch.zeros(pad_count, config.raw_memory_dim, dtype=keep.dtype)
                keep = torch.cat([pad, keep], dim=0)
            mask = torch.zeros(config.memory_tokens, dtype=torch.bool)
            if pad_count:
                mask[:pad_count] = True
            memories.append(keep.to(torch.float16))
            padding_masks.append(mask)
            summaries.append(_summary(keep, mask).to(torch.float16))
            anchor_idx_local = local[anchor_token]
            anchors.append(emb[anchor_idx_local].to(torch.float16))
            anchor_local.append(anchor_idx_local)
            targets.append([local[token] for token in future])

    if not memories:
        raise RuntimeError("no Qwen3-1.7B training examples were constructed")
    return TrainingTensors(
        raw_memory=torch.stack(memories),
        memory_padding_mask=torch.stack(padding_masks),
        context_summary=torch.stack(summaries),
        anchor_embedding=torch.stack(anchors),
        anchor_local=torch.tensor(anchor_local, dtype=torch.long),
        targets_local=torch.tensor(targets, dtype=torch.long),
    )


def _sample_indices(length: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, length, (min(int(batch_size), length),), generator=generator)


def train_bundle(
    config: Qwen17Config,
    candidate_head_weight: torch.Tensor,
    tensors: TrainingTensors,
    *,
    drafter_steps: int = 320,
    selector_steps: int = 160,
    jump_steps: int = 140,
    fused_steps: int = 140,
    batch_size: int = 8,
    seed: int = 7,
) -> tuple[HiddenFusionDrafter, LowRankSelector, JumpHead, FusedJumpHead, dict]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    head = candidate_head_weight.detach().float().contiguous()

    drafter = HiddenFusionDrafter(config, head)
    drafter.train()
    opt = torch.optim.AdamW(drafter.parameters(), lr=1.2e-3, weight_decay=0.01)
    positions = torch.arange(config.speculative_tokens, dtype=torch.float32)
    static_weights = torch.exp(-positions / max(float(config.loss_gamma), 1e-6))
    ema_accuracy = torch.full((config.speculative_tokens,), 0.5, dtype=torch.float32)
    last_draft_loss = 0.0
    for _ in range(max(1, int(drafter_steps))):
        idx = _sample_indices(len(tensors), batch_size, generator)
        logits = drafter(tensors.raw_memory[idx], tensors.memory_padding_mask[idx], tensors.anchor_embedding[idx])
        target = tensors.targets_local[idx]
        per_token = F.cross_entropy(
            logits.reshape(-1, config.candidate_size), target.reshape(-1), reduction="none"
        ).view(target.shape[0], config.speculative_tokens)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == target).float().mean(dim=0)
            ema_accuracy.mul_(0.9).add_(0.1 * acc)
            adaptive = 1.0 + 0.75 * (1.0 - ema_accuracy)
            weights = static_weights * adaptive
            weights = weights / weights.mean()
        loss = (per_token * weights.unsqueeze(0)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
        opt.step()
        last_draft_loss = float(loss.detach())
    drafter.eval()

    selector = LowRankSelector(config)
    selector.train()
    opt_s = torch.optim.AdamW(selector.parameters(), lr=1.5e-3, weight_decay=0.01)
    last_selector_loss = 0.0
    for _ in range(max(1, int(selector_steps))):
        idx = _sample_indices(len(tensors), max(batch_size, 16), generator)
        ctx = tensors.context_summary[idx]
        anc = tensors.anchor_embedding[idx]
        target = tensors.targets_local[idx]
        first = selector.first_logits(ctx, anc)
        loss = F.cross_entropy(first, target[:, 0])
        if config.speculative_tokens > 1:
            prev = target[:, :-1]
            trans = selector.transition_logits(ctx[:, None, :].expand(-1, prev.shape[1], -1), prev)
            loss = loss + F.cross_entropy(trans.reshape(-1, config.candidate_size), target[:, 1:].reshape(-1))
        opt_s.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        opt_s.step()
        last_selector_loss = float(loss.detach())
    selector.eval()

    jump = JumpHead(config)
    jump.train()
    opt_j = torch.optim.AdamW(jump.parameters(), lr=1.3e-3, weight_decay=0.01)
    jump_positions = [min(config.speculative_tokens - 1, max(0, int(offset) - 1)) for offset in config.jump_offsets]
    last_jump_loss = 0.0
    for _ in range(max(1, int(jump_steps))):
        idx = _sample_indices(len(tensors), batch_size, generator)
        hidden = jump(tensors.context_summary[idx], tensors.anchor_embedding[idx])
        logits = torch.einsum("bjh,ch->bjc", hidden, head)
        target = tensors.targets_local[idx][:, jump_positions]
        loss = F.cross_entropy(logits.reshape(-1, config.candidate_size), target.reshape(-1))
        opt_j.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(jump.parameters(), 1.0)
        opt_j.step()
        last_jump_loss = float(loss.detach())
    jump.eval()

    fused = FusedJumpHead(config)
    fused.train()
    opt_f = torch.optim.AdamW(fused.parameters(), lr=1.0e-3, weight_decay=0.01)
    last_fused_loss = 0.0
    for _ in range(max(1, int(fused_steps))):
        idx = _sample_indices(len(tensors), batch_size, generator)
        with torch.no_grad():
            latent, base_future, _ = drafter.hidden_and_logits(
                tensors.raw_memory[idx], tensors.memory_padding_mask[idx], tensors.anchor_embedding[idx]
            )
        hidden = fused(latent, base_future)
        logits = torch.einsum("bjh,ch->bjc", hidden, head)
        target = tensors.targets_local[idx][:, jump_positions]
        loss = F.cross_entropy(logits.reshape(-1, config.candidate_size), target.reshape(-1))
        opt_f.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fused.parameters(), 1.0)
        opt_f.step()
        last_fused_loss = float(loss.detach())
    fused.eval()

    correct = torch.zeros(config.speculative_tokens, dtype=torch.float64)
    count = 0
    selector_correct = torch.zeros(config.speculative_tokens, dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(tensors), 16):
            stop = min(start + 16, len(tensors))
            logits = drafter(tensors.raw_memory[start:stop], tensors.memory_padding_mask[start:stop], tensors.anchor_embedding[start:stop])
            target = tensors.targets_local[start:stop]
            correct += (logits.argmax(dim=-1) == target).double().sum(dim=0)
            ctx = tensors.context_summary[start:stop]
            anc = tensors.anchor_embedding[start:stop]
            selector_correct[0] += (selector.first_logits(ctx, anc).argmax(dim=-1) == target[:, 0]).double().sum()
            for pos in range(1, config.speculative_tokens):
                pred = selector.transition_logits(ctx, target[:, pos - 1]).argmax(dim=-1)
                selector_correct[pos] += (pred == target[:, pos]).double().sum()
            count += int(target.shape[0])

    metrics = {
        "examples": len(tensors),
        "drafter_steps": int(drafter_steps),
        "selector_steps": int(selector_steps),
        "jump_steps": int(jump_steps),
        "fused_steps": int(fused_steps),
        "batch_size": int(batch_size),
        "drafter_final_loss": last_draft_loss,
        "selector_final_loss": last_selector_loss,
        "jump_final_loss": last_jump_loss,
        "fused_final_loss": last_fused_loss,
        "drafter_position_accuracy": [float(x / max(count, 1)) for x in correct.tolist()],
        "selector_teacher_path_accuracy": [float(x / max(count, 1)) for x in selector_correct.tolist()],
        "trainable_parameter_count": int(sum(p.numel() for module in (drafter, selector, jump, fused) for p in module.parameters() if p.requires_grad)),
        "loss_objective": "prefix-weighted drafter CE + context-gated selector CE + sparse jump CE + fused residual CE",
    }
    return drafter, selector, jump, fused, metrics


def save_bundle(
    path: str | Path,
    config: Qwen17Config,
    candidate_ids: list[int],
    drafter: HiddenFusionDrafter,
    selector: LowRankSelector,
    jump: JumpHead,
    fused: FusedJumpHead,
    metadata: dict,
) -> None:
    payload = {
        "format_version": 2,
        "config": {
            **asdict(config),
            "target_layer_ids": list(config.target_layer_ids),
            "jump_offsets": list(config.jump_offsets),
        },
        "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
        "drafter": drafter.state_dict(),
        "selector": selector.state_dict(),
        "jump": jump.state_dict(),
        "fused": fused.state_dict(),
        "metadata": metadata,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_bundle(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw = dict(payload["config"])
    raw["target_layer_ids"] = tuple(int(x) for x in raw["target_layer_ids"])
    raw["jump_offsets"] = tuple(int(x) for x in raw["jump_offsets"])
    config = Qwen17Config(**raw)
    candidate_ids = payload["candidate_ids"].long()
    head = payload["drafter"]["candidate_head_weight"].float()
    drafter = HiddenFusionDrafter(config, head)
    selector = LowRankSelector(config)
    jump = JumpHead(config)
    fused = FusedJumpHead(config)
    drafter.load_state_dict(payload["drafter"])
    selector.load_state_dict(payload["selector"])
    jump.load_state_dict(payload["jump"])
    fused.load_state_dict(payload["fused"])
    for module in (drafter, selector, jump, fused):
        module.eval()
    return config, candidate_ids, drafter, selector, jump, fused, payload.get("metadata", {})
