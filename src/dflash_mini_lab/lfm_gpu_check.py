from __future__ import annotations

import json
import shutil
import subprocess
import sys

import torch


def main() -> None:
    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        report["nvidia_smi"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    else:
        report["nvidia_smi"] = "not present in container PATH; PyTorch CUDA check is authoritative"

    if not torch.cuda.is_available():
        print(json.dumps(report, indent=2))
        print(
            "ERROR: CUDA is not visible. Ensure an NVIDIA driver and NVIDIA Container Toolkit are installed, "
            "then run the Compose service with GPU access.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "compute_capability": f"{p.major}.{p.minor}",
                "total_memory_gib": round(float(p.total_memory) / (1024**3), 2),
            }
        )
    report["devices"] = devices

    # Real CUDA execution check, not only driver discovery.
    a = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    b = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    report["cuda_matmul_ok"] = bool(torch.isfinite(c).all().item())
    report["test_tensor_device"] = str(c.device)

    print(json.dumps(report, indent=2))
    if not report["cuda_matmul_ok"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
