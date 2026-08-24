# 🔬 Reproducibility Guide

Complete instructions to reproduce ALL paper results.

## Prerequisites

### Hardware
- NVIDIA A100 GPU with CUDA 12.9+ (RTX 30/40 series recommended)
- SXM4 40GB
- 16GB+ RAM (32GB for large experiments)
- 100GB+ disk space (for SNAP datasets)

### Software
```bash
# Python 3.9+
Cuda version 12.9
python --version

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"
