from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


BLOCK_SIZE = 4
JUMP_OFFSETS = (2, 4)


@dataclass(frozen=True)
class AuxConfig:
    model_id: str
    target_hidden_size: int
    target_vocab_size: int
    candidate_size: int
    block_size: int = BLOCK_SIZE
    context_tokens: int = 4
    drafter_dim: int = 128
    drafter_heads: int = 4
    selector_rank: int = 16
    jump_dim: int = 48
    fused_rank: int = 16

    @property
    def context_dim(self) -> int:
        # Preserve local word order without a target forward: flatten the most
        # recent frozen input embeddings, then append the prefix mean embedding.
        return self.target_hidden_size * (self.context_tokens + 1)


class CompactParallelDrafter(nn.Module):
    """Small non-causal block drafter over a compact target-token candidate set."""

    def __init__(self, config: AuxConfig):
        super().__init__()
        self.config = config
        self.context_proj = nn.Linear(config.context_dim, config.drafter_dim)
        self.slot_emb = nn.Embedding(config.block_size, config.drafter_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.drafter_dim,
            nhead=config.drafter_heads,
            dim_feedforward=config.drafter_dim * 3,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.block_net = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(config.drafter_dim)
        self.head = nn.Linear(config.drafter_dim, config.candidate_size)

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        slots = torch.arange(self.config.block_size, device=context.device).unsqueeze(0).expand(batch, -1)
        x = self.context_proj(context).unsqueeze(1) + self.slot_emb(slots)
        return self.norm(self.block_net(x))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(context))


class PairwisePathSelector(nn.Module):
    """Low-rank predecessor-conditioned scorer using original LFM token ids."""

    def __init__(self, config: AuxConfig):
        super().__init__()
        rank = config.selector_rank
        self.a = nn.Embedding(config.target_vocab_size, rank)
        self.b = nn.Embedding(config.target_vocab_size, rank)
        self.gate = nn.Sequential(nn.Linear(config.context_dim, rank), nn.Tanh())
        self.scale = rank ** -0.5

    def candidate_correction(
        self,
        prev_tokens: torch.Tensor,
        context: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = self.a(prev_tokens) * self.gate(context).unsqueeze(1)
        code = self.b(candidate_ids)
        return torch.einsum("bpr,cr->bpc", q, code) * self.scale


class CompactJumpHead(nn.Module):
    def __init__(self, config: AuxConfig):
        super().__init__()
        self.offsets = JUMP_OFFSETS
        self.context_proj = nn.Linear(config.context_dim, config.jump_dim)
        self.offset_emb = nn.Embedding(config.block_size + 1, config.jump_dim)
        self.norm = nn.LayerNorm(config.jump_dim)
        self.head = nn.Linear(config.jump_dim, config.candidate_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        offsets = torch.tensor(self.offsets, dtype=torch.long, device=context.device)
        h = self.context_proj(context).unsqueeze(1) + self.offset_emb(offsets).unsqueeze(0)
        return self.head(self.norm(torch.tanh(h)))


class CompactFusedResidual(nn.Module):
    """DFlash4-distilled residual evaluated only on compact candidate ids."""

    def __init__(self, config: AuxConfig):
        super().__init__()
        self.offsets = JUMP_OFFSETS
        self.rank = config.fused_rank
        self.query = nn.Linear(config.drafter_dim, config.fused_rank)
        self.offset_emb = nn.Embedding(config.block_size + 1, config.fused_rank)
        self.codebook = nn.Embedding(config.candidate_size, config.fused_rank)
        self.scale = config.fused_rank ** -0.5

    def residual_logits(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        offsets = torch.tensor(self.offsets, dtype=torch.long, device=draft_hidden.device)
        positions = offsets - 1
        q = torch.tanh(self.query(draft_hidden[:, positions]) + self.offset_emb(offsets).unsqueeze(0))
        return torch.einsum("bjr,cr->bjc", q, self.codebook.weight) * self.scale


@dataclass
class TrainingTensors:
    context: torch.Tensor
    future_local: torch.Tensor
    prev_target: torch.Tensor

    def __len__(self) -> int:
        return int(self.context.shape[0])


def make_training_tensors(
    sequences: list[list[int]],
    embedding_weight: torch.Tensor,
    candidate_ids: list[int],
    *,
    training_start_indices: list[int] | None = None,
    block_size: int = BLOCK_SIZE,
    context_tokens: int = 4,
) -> TrainingTensors:
    """Cache exact teacher-trajectory examples without extra target forwards.

    ``training_start_indices`` is the number of prompt tokens in each generated
    sequence. When supplied, examples begin at the final prompt token, so all
    future labels come from the LFM greedy trajectory rather than human prompt
    text.
    """
    local = {int(tok): i for i, tok in enumerate(candidate_ids)}
    contexts: list[torch.Tensor] = []
    futures: list[list[int]] = []
    prevs: list[list[int]] = []
    emb = embedding_weight.detach().float().cpu()
    starts = training_start_indices or [1] * len(sequences)
    if len(starts) != len(sequences):
        raise ValueError("training_start_indices must match sequences")

    for seq_list, prompt_len in zip(sequences, starts):
        seq = torch.tensor(seq_list, dtype=torch.long)
        if seq.numel() <= block_size + 1:
            continue
        seq_emb = emb[seq]
        cumulative = seq_emb.cumsum(dim=0)
        first_end = max(0, int(prompt_len) - 1)
        for end in range(first_end, int(seq.numel()) - block_size):
            future_target = [int(x) for x in seq[end + 1 : end + 1 + block_size].tolist()]
            if any(tok not in local for tok in future_target):
                continue
            mean = cumulative[end] / float(end + 1)
            recent = seq_emb[max(0, end - context_tokens + 1) : end + 1]
            if int(recent.shape[0]) < context_tokens:
                pad = torch.zeros(context_tokens - int(recent.shape[0]), int(seq_emb.shape[1]), dtype=seq_emb.dtype)
                recent = torch.cat([pad, recent], dim=0)
            contexts.append(torch.cat([recent.reshape(-1), mean], dim=-1))
            futures.append([local[tok] for tok in future_target])
            prevs.append([int(seq[end])] + future_target[:-1])

    if not contexts:
        raise RuntimeError("No LFM auxiliary training examples were constructed")
    return TrainingTensors(
        context=torch.stack(contexts).float(),
        future_local=torch.tensor(futures, dtype=torch.long),
        prev_target=torch.tensor(prevs, dtype=torch.long),
    )


def _batch_indices(size: int, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, size, (min(batch_size, size),), generator=generator)


def train_auxiliary_models(
    config: AuxConfig,
    candidate_ids: list[int],
    tensors: TrainingTensors,
    *,
    seed: int = 7,
    drafter_steps: int = 180,
    selector_steps: int = 100,
    jump_steps: int = 80,
    fused_steps: int = 100,
    batch_size: int = 32,
) -> tuple[CompactParallelDrafter, PairwisePathSelector, CompactJumpHead, CompactFusedResidual, dict]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    candidates = torch.tensor(candidate_ids, dtype=torch.long)

    drafter = CompactParallelDrafter(config)
    selector = PairwisePathSelector(config)
    jump = CompactJumpHead(config)
    fused = CompactFusedResidual(config)

    drafter.train(); opt = torch.optim.AdamW(drafter.parameters(), lr=2e-3); last_draft_loss = 0.0
    for _ in range(max(1, int(drafter_steps))):
        idx = _batch_indices(len(tensors), batch_size, generator)
        logits = drafter(tensors.context[idx])
        loss = F.cross_entropy(logits.reshape(-1, config.candidate_size), tensors.future_local[idx].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); last_draft_loss = float(loss.detach())
    drafter.eval()
    for p in drafter.parameters(): p.requires_grad_(False)

    selector.train(); opt = torch.optim.AdamW(selector.parameters(), lr=1.5e-3); last_selector_loss = 0.0
    for _ in range(max(1, int(selector_steps))):
        idx = _batch_indices(len(tensors), batch_size, generator); context = tensors.context[idx]
        with torch.no_grad(): base = drafter(context)
        corrected = base + selector.candidate_correction(tensors.prev_target[idx], context, candidates)
        loss = F.cross_entropy(corrected.reshape(-1, config.candidate_size), tensors.future_local[idx].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); last_selector_loss = float(loss.detach())
    selector.eval()
    for p in selector.parameters(): p.requires_grad_(False)

    jump.train(); opt = torch.optim.AdamW(jump.parameters(), lr=2e-3); jump_positions = torch.tensor([offset - 1 for offset in JUMP_OFFSETS], dtype=torch.long); last_jump_loss = 0.0
    for _ in range(max(1, int(jump_steps))):
        idx = _batch_indices(len(tensors), batch_size, generator); logits = jump(tensors.context[idx]); targets = tensors.future_local[idx][:, jump_positions]
        loss = F.cross_entropy(logits.reshape(-1, config.candidate_size), targets.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); last_jump_loss = float(loss.detach())
    jump.eval()
    for p in jump.parameters(): p.requires_grad_(False)

    fused.train(); opt = torch.optim.AdamW(fused.parameters(), lr=1.5e-3); last_fused_loss = 0.0
    for _ in range(max(1, int(fused_steps))):
        idx = _batch_indices(len(tensors), batch_size, generator); context = tensors.context[idx]
        with torch.no_grad():
            hidden = drafter.encode(context); base = drafter.head(hidden)[:, jump_positions]; teacher = base + 0.5 * jump(context); teacher_probs = F.softmax(teacher.reshape(-1, config.candidate_size), dim=-1)
        student = base + fused.residual_logits(hidden); flat = student.reshape(-1, config.candidate_size); targets = tensors.future_local[idx][:, jump_positions].reshape(-1)
        ce = F.cross_entropy(flat, targets); kl = F.kl_div(F.log_softmax(flat, dim=-1), teacher_probs, reduction="batchmean"); loss = 0.7 * ce + 0.3 * kl
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); last_fused_loss = float(loss.detach())
    fused.eval()

    metrics = {
        "examples": len(tensors), "drafter_steps": int(drafter_steps), "selector_steps": int(selector_steps), "jump_steps": int(jump_steps), "fused_steps": int(fused_steps),
        "final_drafter_loss": last_draft_loss, "final_selector_loss": last_selector_loss, "final_jump_loss": last_jump_loss, "final_fused_loss": last_fused_loss,
        "aux_parameter_count": sum(p.numel() for model in (drafter, selector, jump, fused) for p in model.parameters()),
    }
    return drafter, selector, jump, fused, metrics


def save_aux_bundle(output_path: str | Path, config: AuxConfig, candidate_ids: list[int], drafter: CompactParallelDrafter, selector: PairwisePathSelector, jump: CompactJumpHead, fused: CompactFusedResidual, metadata: dict) -> None:
    payload = {"format_version": 2, "config": asdict(config), "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long), "drafter": drafter.state_dict(), "selector": selector.state_dict(), "jump": jump.state_dict(), "fused": fused.state_dict(), "metadata": metadata}
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)


def load_aux_bundle(path: str | Path):
    payload = torch.load(path, map_location="cpu", weights_only=False); config = AuxConfig(**payload["config"]); candidate_ids = payload["candidate_ids"].long()
    drafter = CompactParallelDrafter(config); drafter.load_state_dict(payload["drafter"]); drafter.eval()
    selector = PairwisePathSelector(config); selector.load_state_dict(payload["selector"]); selector.eval()
    jump = CompactJumpHead(config); jump.load_state_dict(payload["jump"]); jump.eval()
    fused = CompactFusedResidual(config); fused.load_state_dict(payload["fused"]); fused.eval()
    return config, candidate_ids, drafter, selector, jump, fused, payload.get("metadata", {})
