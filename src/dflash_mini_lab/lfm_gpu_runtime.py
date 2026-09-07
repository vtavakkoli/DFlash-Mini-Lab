from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .lfm_dspark import LfmDSparkRuntime


class _CudaDrafterAdapter:
    """Keep the existing drafter API while moving CPU inputs onto CUDA."""

    def __init__(self, module: torch.nn.Module, device: torch.device):
        self.module = module
        self.device = device

    @property
    def head(self):
        return self.module.head

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.module.encode(x.to(self.device, non_blocking=False))

    def __call__(self, x: torch.Tensor):
        return self.module(x.to(self.device, non_blocking=False))

    def eval(self):
        self.module.eval()
        return self

    def parameters(self):
        return self.module.parameters()

    def __getattr__(self, name):
        if name in {"module", "device"}:
            raise AttributeError(name)
        return getattr(self.module, name)


class LfmGpuRuntime(LfmDSparkRuntime):
    """GPU-accelerated LFM2.5 runtime for the All-14 benchmark.

    The expensive LFM target and DFlash drafter execute on CUDA. Small selector,
    MOBS, DSpark and Parareal guidance remains on CPU/NumPy so the implementation
    preserves the existing method logic and includes host/device transfer cost.
    """

    last_device_info: dict | None = None

    def __init__(
        self,
        aux_path: str | Path,
        dspark_path: str | Path,
        *,
        model_id: str | None = None,
        cpu_threads: int | None = None,
        dtype: str = "float16",
        cuda_device: int | None = None,
        **kwargs,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available inside the container. Install the NVIDIA driver + "
                "NVIDIA Container Toolkit and start this service with GPU access."
            )

        # Load frozen artifacts in their existing CPU-compatible representation.
        # DSpark/selector arrays are materialized before the expensive modules are
        # moved to CUDA.
        super().__init__(
            aux_path,
            dspark_path,
            model_id=model_id,
            cpu_threads=cpu_threads,
            dtype="float32",
            **kwargs,
        )

        index = int(os.getenv("CUDA_DEVICE", str(0 if cuda_device is None else cuda_device)))
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is invalid; found {torch.cuda.device_count()} device(s)")
        self.device = torch.device(f"cuda:{index}")

        dtype_name = str(os.getenv("LFM_GPU_DTYPE", dtype)).lower()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported GPU dtype {dtype_name!r}; use float16, bfloat16 or float32")
        self.gpu_dtype_name = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[dtype_map[dtype_name]]
        self.gpu_dtype = dtype_map[dtype_name]

        torch.cuda.set_device(self.device)
        torch.set_float32_matmul_precision("high")
        if os.getenv("LFM_ALLOW_TF32", "1") not in {"0", "false", "False"}:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Target dominates runtime, so it is the primary CUDA resident.
        self.target.to(device=self.device, dtype=self.gpu_dtype)
        self.target.eval()
        self.gpu_embedding = self.target.get_input_embeddings().weight.detach()

        # V12 and a few legacy selectors index runtime.embedding from CPU code.
        # Keep a CPU float32 copy for those tiny top-k lookups, while context
        # construction uses gpu_embedding and therefore stays CUDA-accelerated.
        self.embedding = self.gpu_embedding.float().cpu()

        # The DFlash drafter runs once per speculative block. Move it to CUDA;
        # the adapter makes legacy CPU-created input tensors work unchanged.
        drafter_module = self.drafter.to(self.device).eval()
        self.drafter = _CudaDrafterAdapter(drafter_module, self.device)

        # selector/jump/fused/DSpark stay CPU-side intentionally. Their measured
        # guidance cost remains visible and their outputs are tiny compared with
        # the target model execution.
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(self.device)
        info = {
            "backend": "cuda",
            "device_index": index,
            "device_name": torch.cuda.get_device_name(self.device),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": int(props.total_memory),
            "target_dtype": self.gpu_dtype_name,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": int(torch.backends.cudnn.version() or 0),
            "target_on_gpu": True,
            "drafter_on_gpu": True,
            "guidance_on_cpu": True,
            "cpu_embedding_copy_for_selectors": True,
        }
        self.device_info = info
        type(self).last_device_info = info

    @torch.inference_mode()
    def target_logits(self, input_ids: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).long().unsqueeze(0).to(self.device)
        out = self.target(input_ids=ids, use_cache=False, return_dict=True)
        # The copy synchronizes CUDA, so existing wall-clock timing remains valid.
        return out.logits[0].float().cpu().numpy()

    @torch.inference_mode()
    def context_features(self, input_ids: np.ndarray) -> np.ndarray:
        ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).long().to(self.device)
        emb = self.gpu_embedding.index_select(0, ids).float()
        n = int(self.config.context_tokens)
        recent = emb[max(0, int(emb.shape[0]) - n) :]
        if int(recent.shape[0]) < n:
            pad = torch.zeros(
                n - int(recent.shape[0]),
                int(emb.shape[1]),
                dtype=emb.dtype,
                device=self.device,
            )
            recent = torch.cat([pad, recent], dim=0)
        context = torch.cat([recent.reshape(-1), emb.mean(dim=0)], dim=-1)
        return context.cpu().numpy().astype(np.float32, copy=False)

    @torch.inference_mode()
    def draft_hidden_and_logits(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(np.asarray(context, dtype=np.float32)).unsqueeze(0).to(self.device)
        hidden = self.drafter.encode(x)
        logits = self.drafter.head(hidden)
        return hidden[0].float().cpu().numpy(), logits[0].float().cpu().numpy()

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)
