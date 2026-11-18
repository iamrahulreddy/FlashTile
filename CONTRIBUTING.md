# Contributing to FlashTile

Contributions are welcome! This document outlines the process for contributing code, documentation, or reporting issues.

## Development Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.4+ (tested on 2.15 with Triton 3.9)
- An NVIDIA GPU for the Triton path; everything else runs on CPU

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iamrahulreddy/FlashTile.git
   cd FlashTile
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev,hf]"
   ```

4. **Verify installation:**
   ```bash
   python -c "import flashtile; print(flashtile.__version__)"
   ```

## How the code fits together

`flashtile/contract.py` decides what attention computes (shapes, GQA mapping, causal positions,
fully masked rows). `reference.py`, `tiled.py` and `triton_kernels.py` all have to agree with it.
If you change one of them, add a test that compares it with the fp64 reference: output, lse and
all three gradients.

## Testing

```bash
# CPU: oracle, tiled path, gradients, gradcheck, contract errors
pytest -v

# One file
pytest tests/test_attention.py -v
```

GPU tests (`tests/test_cuda.py`, `test_triton.py`, `test_hf.py`) skip themselves without CUDA. Run
them on a real GPU, locally with `pytest` or on Modal:

```bash
modal run modal_app.py --gpus H200 --skip-bench --skip-llm
modal run modal_app.py --gpus H200 --skip-bench --skip-llm --pytest-args "tests/test_triton.py -k decode"
```

### Pre-Submission Checklist

```bash
# 1. Imports and backend selection
python -c "import flashtile, torch; q = torch.randn(1, 2, 8, 4); print(flashtile.select_backend(q, q, q))"

# 2. CPU suite
pytest -v

# 3. GPU suite (if you changed anything under flashtile/)
modal run modal_app.py --gpus H200 --skip-bench --skip-llm
```

For a performance change, also run the benchmark on the GPUs you claim it for:
`modal run modal_app.py --gpus H200 --skip-tests --skip-llm`. Mention the rows that got slower too.

## Project Structure

```
flashtile/
├── contract.py         # input validation and the operation contract
├── reference.py        # dense fp64-capable oracle
├── tiled.py            # tiled PyTorch forward/backward (any device, dtype, mask)
├── triton_kernels.py   # fused Triton forward/backward
├── api.py              # attention(), backend selection
├── hf.py               # Hugging Face transformers integration
└── __init__.py         # public API

tests/                  # CPU and GPU test suites
benchmark/              # kernel benchmark, validation, Qwen3.5 evaluation, Qwen3.8-27B comparison, reports
docs/                   # USAGE.md, PERFORMANCE.md and the cross-GPU dashboard
flashtile_results_*/    # published results, one folder per GPU
modal_app.py            # runs everything on Modal GPUs
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/description`
3. Make your changes
4. **Run the test suite** and ensure all tests pass
5. Update documentation if needed
6. Submit a pull request with clear description

## Reporting Issues

When reporting bugs, please include:
- Python version (`python --version`)
- PyTorch and Triton versions (`python -c "import torch, triton; print(torch.__version__, triton.__version__)"`)
- GPU model (if applicable): `nvidia-smi`
- The backend that ran: `flashtile.select_backend(q, k, v, ...)`
- Minimal code to reproduce the issue
- Full error traceback
- FlashTile version (`python -c "import flashtile; print(flashtile.__version__)"`)

Thank you for contributing!
