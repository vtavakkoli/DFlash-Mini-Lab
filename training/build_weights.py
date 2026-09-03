from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SEED = 7
BLOCK_SIZE = 4
JUMP_OFFSETS = (2, 4)
TOKEN_RE = re.compile(r"<eos>|[A-Za-z0-9]+|[^\w\s]")
SENTENCES = ["one two three four five six seven eight nine ten .","the capital of austria is vienna .","vienna is the capital of austria .","the capital of france is paris .","the capital of italy is rome .","the quick brown fox jumps over the lazy dog .","machine learning models predict the next token .","language models generate text one token at a time .","speculative decoding proposes tokens then verifies them .","a draft model proposes several future tokens .","the target model verifies the proposed token block .","accepted draft tokens reduce expensive target model calls .","dflash predicts a draft block in parallel .","parallel drafting removes sequential draft latency .","dflash two keeps several candidates at every position .","a path selector chooses a coherent candidate sequence .","the verifier keeps correct tokens and fixes the first mismatch .","greedy speculative decoding can match greedy autoregressive decoding .","small models are useful for educational experiments .","benchmark speed quality acceptance and target forward passes ."]


def make_corpus(repeats: int = 40) -> str:
    return " <eos> ".join(SENTENCES * repeats) + " <eos>"


@dataclass
class WordTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_text(cls, text: str) -> "WordTokenizer":
        toks = TOKEN_RE.findall(text.lower()); vocab = ["<pad>", "<bos>", "<unk>"] + sorted(set(toks)); return cls({t: i for i, t in enumerate(vocab)}, vocab)

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = [self.stoi.get(t, self.stoi["<unk>"]) for t in TOKEN_RE.findall(text.lower())]; return ([self.stoi["<bos>"]] + ids) if add_bos else ids


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 96, nhead: int = 4, layers: int = 2, max_len: int = 256):
        super().__init__(); self.d_model = d_model; self.token_emb = nn.Embedding(vocab_size, d_model); self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.0, batch_first=True, norm_first=True); self.transformer = nn.TransformerEncoder(layer, num_layers=layers); self.norm = nn.LayerNorm(d_model); self.lm_head = nn.Linear(d_model, vocab_size, bias=False); self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape; pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0); x = self.token_emb(input_ids) + self.pos_emb(pos); causal = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1); return self.lm_head(self.norm(self.transformer(x, mask=causal)))

    @torch.no_grad()
    def cheap_context_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.token_emb(input_ids); return torch.cat([emb[:, -1, :], emb.mean(dim=1)], dim=-1)


class ParallelBlockDrafter(nn.Module):
    def __init__(self, vocab_size: int, target_dim: int, block_size: int = 4, d_model: int = 64, nhead: int = 4):
        super().__init__(); self.d_model = d_model; self.context_proj = nn.Linear(target_dim * 2, d_model); self.slot_emb = nn.Embedding(block_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 3, dropout=0.0, batch_first=True, norm_first=True); self.block_net = nn.TransformerEncoder(layer, num_layers=1); self.norm = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, vocab_size)

    def encode(self, context: torch.Tensor) -> torch.Tensor:
        bsz = context.size(0); slots = torch.arange(self.slot_emb.num_embeddings, device=context.device).unsqueeze(0).expand(bsz, -1); x = self.context_proj(context).unsqueeze(1) + self.slot_emb(slots); return self.norm(self.block_net(x))

    def forward(self, context: torch.Tensor) -> torch.Tensor: return self.head(self.encode(context))


class PairwisePathSelector(nn.Module):
    def __init__(self, vocab_size: int, context_dim: int, rank: int = 32):
        super().__init__(); self.a = nn.Embedding(vocab_size, rank); self.b = nn.Embedding(vocab_size, rank); self.gate = nn.Sequential(nn.Linear(context_dim, rank), nn.Tanh()); self.scale = rank ** -0.5

    def correction_logits(self, prev_tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.a(prev_tokens) * self.gate(context).unsqueeze(1); return torch.einsum("bkr,vr->bkv", q, self.b.weight) * self.scale


class JumpAnchorHead(nn.Module):
    def __init__(self, vocab_size: int, context_dim: int, offsets: tuple[int, ...] = JUMP_OFFSETS, d_model: int = 48):
        super().__init__(); self.offsets = tuple(int(x) for x in offsets); self.context_proj = nn.Linear(context_dim, d_model); self.offset_emb = nn.Embedding(BLOCK_SIZE + 1, d_model); self.norm = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        offsets = torch.tensor(self.offsets, dtype=torch.long, device=context.device); h = self.context_proj(context).unsqueeze(1) + self.offset_emb(offsets).unsqueeze(0); return self.head(self.norm(torch.tanh(h)))


class FusedJumpResidual(nn.Module):
    """DFlash5 residual distilled from DFlash4 but evaluated only on top-k at inference."""
    def __init__(self, vocab_size: int, draft_dim: int, offsets: tuple[int, ...] = JUMP_OFFSETS, rank: int = 32):
        super().__init__(); self.offsets = tuple(int(x) for x in offsets); self.rank = rank; self.query = nn.Linear(draft_dim, rank); self.offset_emb = nn.Embedding(BLOCK_SIZE + 1, rank); self.codebook = nn.Embedding(vocab_size, rank); self.scale = rank ** -0.5

    def queries(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        offsets = torch.tensor(self.offsets, dtype=torch.long, device=draft_hidden.device); positions = offsets - 1; h = draft_hidden[:, positions, :]; return torch.tanh(self.query(h) + self.offset_emb(offsets).unsqueeze(0))

    def full_residual_logits(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bjr,vr->bjv", self.queries(draft_hidden), self.codebook.weight) * self.scale


def random_lm_batch(tokens, batch_size, seq_len):
    starts = torch.randint(0, tokens.numel() - seq_len - 1, (batch_size,)); x = torch.stack([tokens[s:s+seq_len] for s in starts.tolist()]); y = torch.stack([tokens[s+1:s+seq_len+1] for s in starts.tolist()]); return x, y


def random_prefix_future_batch(tokens, batch_size, prefix_len, block_size):
    starts = torch.randint(0, tokens.numel() - prefix_len - block_size - 1, (batch_size,)); prefix = torch.stack([tokens[s:s+prefix_len] for s in starts.tolist()]); future = torch.stack([tokens[s+prefix_len:s+prefix_len+block_size] for s in starts.tolist()]); return prefix, future


def build(output_dir: Path, target_steps: int, draft_steps: int, selector_steps: int, jump_steps: int, fused_jump_steps: int) -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    text = make_corpus(); tok = WordTokenizer.from_text(text); target = TinyTransformerLM(len(tok.itos)); drafter = ParallelBlockDrafter(len(tok.itos), target_dim=target.d_model, block_size=BLOCK_SIZE); selector = PairwisePathSelector(len(tok.itos), context_dim=target.d_model * 2); jump = JumpAnchorHead(len(tok.itos), context_dim=target.d_model * 2); fused_jump = FusedJumpResidual(len(tok.itos), draft_dim=drafter.d_model); ids = torch.tensor(tok.encode(text, add_bos=True), dtype=torch.long)

    target.train(); opt = torch.optim.AdamW(target.parameters(), lr=2e-3)
    for _ in range(target_steps):
        x,y=random_lm_batch(ids,24,24); logits=target(x); loss=F.cross_entropy(logits.reshape(-1,len(tok.itos)),y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    target.eval(); [p.requires_grad_(False) for p in target.parameters()]

    drafter.train(); opt=torch.optim.AdamW(drafter.parameters(),lr=2e-3)
    for _ in range(draft_steps):
        prefix,future=random_prefix_future_batch(ids,32,14,BLOCK_SIZE)
        with torch.no_grad(): context=target.cheap_context_features(prefix)
        logits=drafter(context); loss=F.cross_entropy(logits.reshape(-1,len(tok.itos)),future.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    drafter.eval(); [p.requires_grad_(False) for p in drafter.parameters()]

    selector.train(); opt=torch.optim.AdamW(selector.parameters(),lr=1.5e-3)
    for _ in range(selector_steps):
        prefix,future=random_prefix_future_batch(ids,32,14,BLOCK_SIZE)
        with torch.no_grad(): context=target.cheap_context_features(prefix); draft_logits=drafter(context)
        prev=torch.cat([prefix[:,-1:],future[:,:-1]],dim=1); corrected=draft_logits+selector.correction_logits(prev,context); loss=F.cross_entropy(corrected.reshape(-1,len(tok.itos)),future.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    selector.eval(); [p.requires_grad_(False) for p in selector.parameters()]

    jump.train(); opt=torch.optim.AdamW(jump.parameters(),lr=2e-3); jump_targets = torch.tensor([offset - 1 for offset in JUMP_OFFSETS], dtype=torch.long)
    for _ in range(jump_steps):
        prefix,future=random_prefix_future_batch(ids,32,14,BLOCK_SIZE)
        with torch.no_grad(): context=target.cheap_context_features(prefix)
        logits=jump(context); target_future=future[:, jump_targets]; loss=F.cross_entropy(logits.reshape(-1,len(tok.itos)),target_future.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    jump.eval(); [p.requires_grad_(False) for p in jump.parameters()]

    # Distill the stronger DFlash4 anchor distribution into a shared-state residual.
    # The teacher is used only at build time. DFlash5 inference never executes it.
    fused_jump.train(); opt=torch.optim.AdamW(fused_jump.parameters(),lr=1.5e-3)
    for _ in range(fused_jump_steps):
        prefix,future=random_prefix_future_batch(ids,32,14,BLOCK_SIZE)
        with torch.no_grad():
            context=target.cheap_context_features(prefix); hidden=drafter.encode(context); draft_logits=drafter.head(hidden); teacher_jump=jump(context)
            teacher_combined=draft_logits[:,jump_targets,:] + 0.5 * teacher_jump
            teacher_probs=F.softmax(teacher_combined.reshape(-1,len(tok.itos)),dim=-1)
        residual=fused_jump.full_residual_logits(hidden); student_combined=draft_logits[:,jump_targets,:] + residual; target_future=future[:,jump_targets]
        flat_student=student_combined.reshape(-1,len(tok.itos)); ce=F.cross_entropy(flat_student,target_future.reshape(-1)); distill=F.kl_div(F.log_softmax(flat_student,dim=-1),teacher_probs,reduction="batchmean"); loss=0.65*ce+0.35*distill
        opt.zero_grad(); loss.backward(); opt.step()
    fused_jump.eval()

    output_dir.mkdir(parents=True,exist_ok=True); arrays={}
    for section,model in (("target",target),("drafter",drafter),("selector",selector),("jump",jump),("fused_jump",fused_jump)):
        for name,value in model.state_dict().items(): arrays[f"{section}.{name}"]=value.detach().cpu().numpy().astype(np.float32)
    arrays["jump_offsets"] = np.asarray(JUMP_OFFSETS, dtype=np.int64); np.savez_compressed(output_dir/"tiny_dflash_lab.npz",**arrays); (output_dir/"tokenizer.json").write_text(json.dumps({"stoi":tok.stoi,"itos":tok.itos},indent=2),encoding="utf-8")
    manifest={"name":"tiny-dflash-cpu-reference","seed":SEED,"torch_builder_version":torch.__version__,"target":{"type":"causal Transformer","layers":2,"hidden_size":96,"heads":4},"drafter":{"type":"non-causal parallel block Transformer","layers":1,"hidden_size":64,"heads":4,"block_size":BLOCK_SIZE},"selector":{"type":"low-rank predecessor-conditioned selector","rank":32},"jump":{"type":"indexed sparse future-token MLP head","hidden_size":48,"offsets":list(JUMP_OFFSETS)},"fused_jump":{"type":"DFlash4-distilled shared-drafter candidate residual","rank":32,"offsets":list(JUMP_OFFSETS),"teacher_weight":0.5},"training_steps":{"target":target_steps,"drafter":draft_steps,"selector":selector_steps,"jump":jump_steps,"fused_jump":fused_jump_steps},"vocab_size":len(tok.itos),"weights_format":"NumPy NPZ float32"}; (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(f"wrote {output_dir/'tiny_dflash_lab.npz'}")


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="models"); p.add_argument("--target-steps",type=int,default=500); p.add_argument("--draft-steps",type=int,default=450); p.add_argument("--selector-steps",type=int,default=250); p.add_argument("--jump-steps",type=int,default=250); p.add_argument("--fused-jump-steps",type=int,default=400); args=p.parse_args(); build(Path(args.output_dir),args.target_steps,args.draft_steps,args.selector_steps,args.jump_steps,args.fused_jump_steps)


if __name__ == "__main__": main()
