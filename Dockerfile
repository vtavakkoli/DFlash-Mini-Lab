FROM python:3.12-slim-bookworm AS model-builder

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /build
COPY training ./training
RUN python -m pip install --upgrade pip \
 && python -m pip install numpy==2.3.5 \
 && python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
 && python training/build_weights.py --output-dir /build/models

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CPU_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .
COPY --from=model-builder /build/models ./models
COPY benchmarks ./benchmarks
RUN mkdir -p /app/reports

# Keep the benchmark container write-compatible with bind-mounted report folders
# across Linux/macOS/Windows hosts. It does not expose a network service.
ENTRYPOINT ["python", "-m", "dflash_mini_lab.cli"]
CMD ["--output-dir", "/app/reports"]
