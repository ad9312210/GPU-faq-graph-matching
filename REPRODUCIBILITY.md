# 🔬 Reproducibility Guide

Complete instructions to reproduce ALL paper results.

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA 11+ (RTX 30/40 series recommended)
- 16GB+ RAM (32GB for large experiments)
- 100GB+ disk space (for SNAP datasets)

### Software
```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)['name'])"