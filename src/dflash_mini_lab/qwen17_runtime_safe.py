from __future__ import annotations

import torch

from .qwen17_runtime import Qwen17Runtime as _Qwen17Runtime


class Qwen17Runtime(_Qwen17Runtime):
    """Tail-safe runtime wrapper for fixed-offset fused anchors.

    DFlash5 has learned +2/+4 fused anchors. Near the end of generation the
    verifier may need fewer than four draft positions, while the fused head
    still owns both fixed offset slots. Pad only the auxiliary latent tensors
    before the fused head; candidate selection remains restricted to the real
    remaining draft logits, so padded positions can never be proposed.
    """

    @torch.inference_mode()
    def _fused_path(self, latent, base_future, logits, summary, anchor_emb, top_k: int, fused_weight: float):
        needed = max(self.fused.positions, default=-1) + 1
        if int(latent.shape[0]) < int(needed):
            pad = int(needed) - int(latent.shape[0])
            latent = torch.cat(
                [latent, torch.zeros(pad, latent.shape[-1], dtype=latent.dtype)],
                dim=0,
            )
            base_future = torch.cat(
                [base_future, torch.zeros(pad, base_future.shape[-1], dtype=base_future.dtype)],
                dim=0,
            )
        return super()._fused_path(latent, base_future, logits, summary, anchor_emb, top_k, fused_weight)
