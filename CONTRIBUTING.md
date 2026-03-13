# Contributing to FlashTile

Contributions are welcome! This document outlines the process for contributing code, documentation, or reporting issues.

## Development Setup

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for testing)

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
   pip install -e ".[all]"
   ```

4. **Verify installation:**
   ```bash
   python -c "from flashtile import FlashAttentionV2; print('OK')"
   ```

## Code Style

### Python Guidelines
- Follow PEP 8
- Use type hints for function signatures
- Write docstrings for public functions (Google style)
- Maximum line length: 100 characters

### Naming Conventions
- Variables: `lowercase_with_underscores`
- Classes: `PascalCase`
- Constants: `UPPERCASE_WITH_UNDERSCORES`

## Testing

FlashTile uses pytest for testing. Run the full test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_correctness.py -v

# Run correctness tests only (CPU-friendly)
pytest tests/test_correctness.py -v -m "not gpu"

# Run with coverage
pytest tests/ --cov=flashtile --cov-report=html
```

### Pre-Submission Checklist

Before submitting changes:

```bash
# 1. Verify imports work
python -c "from flashtile import NaiveAttention, FlashAttentionV1, FlashAttentionV2"

# 2. Run correctness tests
pytest tests/test_correctness.py -v

# 3. Run integration tests
pytest tests/test_integration.py -v

# 4. Quick sanity check (if you have GPU)
python -c "
import torch
from flashtile import NaiveAttention, FlashAttentionV1
x = torch.randn(1, 128, 512)
naive = NaiveAttention(512, 8)
flash = FlashAttentionV1(512, 8)
diff = (naive(x)[0] - flash(x)[0]).abs().max()
assert diff < 0.01, f'Error too large: {diff}'
print('Correctness test passed!')
"
```

## Project Structure

```
flashtile/
├── attention/          # Attention implementations
│   ├── base_attention.py
│   ├── naive_attention.py
│   ├── flash_attention_v1.py
│   ├── flash_attention_v2.py
│   ├── grouped_query_attention.py
│   ├── sliding_window_attention.py
│   └── masked_attention.py
├── kernels/            # GPU kernels
│   └── triton_flash_kernel.py
├── utils/              # Utilities
│   ├── memory_profiler.py
│   ├── kernel_profiler.py
│   ├── attention_visualizer.py
│   └── visualization.py
└── __init__.py         # Public API

tests/                  # Test suite
benchmark/              # Benchmarking tools
demo/                   # Interactive demos
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
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- GPU model (if applicable): `nvidia-smi`
- Minimal code to reproduce the issue
- Full error traceback
- FlashTile version (`python -c "from flashtile import __version__; print(__version__)"`)

## Development Tips

### Adding New Attention Variants

1. Create new file in `flashtile/attention/`
2. Inherit from `BaseAttention`
3. Implement `forward()` and `get_memory_usage()`
4. Add to `flashtile/__init__.py` exports
5. Add tests in `tests/`
6. Update documentation

### Code Quality Tools

```bash
# Format code
black flashtile/ tests/

# Sort imports
isort flashtile/ tests/

# Type checking (optional)
mypy flashtile/
```

Thank you for contributing!
