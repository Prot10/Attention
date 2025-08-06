# Attention Research Framework

A production-ready research framework for comparing attention mechanisms in PyTorch and JAX. This repository provides implementations of various attention mechanisms, training utilities, profiling tools, and comparison scripts for ML research.

## 🚀 Quick Start

1. **Install dependencies:**

   ```bash
   uv sync
   ```

1. **Set up Weights & Biases (optional but recommended):**

   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your wandb API key from https://wandb.ai/settings
   # WANDB_API_KEY=your_actual_api_key_here

   # Test your setup
   uv run python scripts/check_env.py
   ```

1. **Run a quick test:**

   ```bash
   uv run python scripts/test_training.py --framework torch
   ```

## 🎯 Goals

- **Single Dataset**: Uses Hugging Face `Salesforce/wikitext` subset `wikitext-103-raw-v1`
- **Custom Tokenizer**: Train your own BPE tokenizer from scratch
- **Multiple Attention Mechanisms**: Implement and compare various attention types BY HAND
- **Dual Framework Support**: Both PyTorch and JAX implementations
- **Performance Analysis**: Compare speed, memory usage, and accuracy
- **Profiling**: Built-in kernel and end-to-end profiling tools
- **CUDA Support**: Optimized for A100 40GB GPUs
- **Single-Host Training**: No multi-node complexity

## 🏗️ Project Structure

```
├── config/
│   └── base.yaml              # Main configuration file
├── data/                      # Data storage (auto-created)
├── src/attention_research/
│   ├── shared/                # Shared utilities
│   │   └── config.py         # Configuration management
│   ├── models/               # Model implementations
│   │   ├── torch/            # PyTorch attention mechanisms
│   │   │   └── attention.py  # Base attention classes
│   │   └── jax/              # JAX/Flax attention mechanisms
│   │       └── attention.py  # Base attention classes
│   ├── data/                 # Data utilities
│   │   ├── __init__.py       # Main data API
│   │   ├── tokenizer.py      # Tokenizer creation and loading
│   │   ├── datasets.py       # Dataset classes and data loaders
│   │   └── utils.py          # Data download and verification
│   │   └── __init__.py       # Dataset and tokenizer utils
│   ├── training/             # Training utilities
│   │   ├── torch_trainer.py  # PyTorch trainer
│   │   └── jax_trainer.py    # JAX trainer
│   └── profiling/            # Profiling tools
├── scripts/
│   ├── download_data.py      # Download WikiText dataset
│   ├── create_datasets.py    # Create and test datasets
│   └── example_training.py   # Example usage script
└── notebooks/                # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### 1. Environment Setup

This project uses `uv` for dependency management:

```bash
# Clone the repository
git clone <repository-url>
cd Attention

# Create and activate environment
uv sync

# This automatically installs:
# - PyTorch 2.1+ with CUDA support
# - JAX with CUDA support
# - Flax, Optax (JAX ecosystem)
# - HuggingFace libraries (datasets, tokenizers, transformers)
# - ML utilities (numpy, scipy, matplotlib, seaborn, pandas)
# - Logging and config (wandb, omegaconf, hydra-core)
# - Development tools (pytest, ruff, mypy)
```

### 2. Download Data

```bash
# Download WikiText-103-raw-v1 dataset
uv run python scripts/download_data.py

# The script will:
# - Download the dataset to ./data/cache/
# - Verify data integrity
# - Show dataset statistics
```

### 3. Configure the Framework

Edit `config/base.yaml` to choose your setup:

```yaml
# Framework selection: 'torch' or 'jax'
framework: torch

# Attention mechanism type
model:
  attention_type: vanilla  # Options: vanilla, multi_head, flash, linear, performer, longformer

# Training parameters
training:
  batch_size: 32
  learning_rate: 5e-4
  max_steps: 50000

# Hardware settings
hardware:
  device: cuda
  mixed_precision: true
  compile_model: true
```

### 4. Test the Setup

```bash
# Run the example script to verify everything works
uv run python scripts/example_training.py

# Create and test datasets
uv run python scripts/create_datasets.py
```

## 🧠 Attention Mechanisms

The framework provides base implementations that you can extend:

### PyTorch Implementations (`src/attention_research/models/torch/`)

- **BaseAttention**: Abstract base class for all attention mechanisms
- **VanillaAttention**: Standard scaled dot-product attention
- **MultiHeadAttention**: Multi-head attention with separate head processing

### JAX/Flax Implementations (`src/attention_research/models/jax/`)

- **BaseAttention**: Abstract base class for JAX attention mechanisms
- **VanillaAttention**: JAX implementation of scaled dot-product attention
- **MultiHeadAttention**: JAX multi-head attention implementation

### Planned Mechanisms (for you to implement)

- **FlashAttention**: Memory-efficient attention
- **LinearAttention**: Linear complexity attention
- **PerformerAttention**: FAVOR+ mechanism
- **LongformerAttention**: Sliding window + global attention

## 🏃‍♂️ Training

### PyTorch Training

```python
from attention_research.training.torch_trainer import TorchTrainer
from attention_research.shared.config import Config

# Load config and model
config = Config()
model = YourAttentionModel(config.get_model_config())

# Create trainer
trainer = TorchTrainer(model, config, train_loader, val_loader)

# Train
trainer.train()
```

### JAX Training

```python
from attention_research.training.jax_trainer import JaxTrainer
from attention_research.shared.config import Config

# Load config and model
config = Config()
model = YourJAXModel()

# Create trainer
trainer = JaxTrainer(model, config, train_data, val_data)

# Train
trainer.train()
```

## 📊 Configuration Management

The framework uses a centralized configuration system:

```python
from attention_research.shared.config import Config

# Load configuration
config = Config()

# Switch frameworks dynamically
config.set_framework("jax")
config.set_attention_type("multi_head")

# Access nested configs
model_config = config.get_model_config()
training_config = config.get_training_config()

# Update configurations
config.update({"training.batch_size": 64})
```

## 🔧 Development Guidelines

### Adding New Attention Mechanisms

1. **PyTorch**: Extend `BaseAttention` in `src/attention_research/models/torch/`
1. **JAX**: Extend `BaseAttention` in `src/attention_research/models/jax/`
1. **Configuration**: Add the new type to valid options in config validation
1. **Testing**: Add tests in the `tests/` directory

### Code Quality

The project includes comprehensive linting and formatting:

```bash
# Run linting
uv run ruff check

# Run type checking
uv run mypy src/

# Run tests
uv run pytest

# Auto-format code
uv run ruff format
```

## 📈 Performance Profiling

The framework includes built-in profiling capabilities:

- **Memory profiling**: Track GPU/CPU memory usage
- **Kernel profiling**: Analyze CUDA kernel performance
- **End-to-end timing**: Measure total training time
- **Attention analysis**: Visualize attention patterns

Enable profiling in `config/base.yaml`:

```yaml
profiling:
  enabled: true
  profile_memory: true
  profile_kernels: true
  trace_activities: ["cpu", "cuda"]
```

## 🎓 Research Usage

This framework is designed for:

- **Attention Mechanism Research**: Compare different attention types
- **Framework Comparison**: PyTorch vs JAX performance analysis
- **Educational Purposes**: Understanding attention implementations
- **Benchmarking**: Standardized comparison methodology

## 📝 Next Steps

1. **Implement Additional Mechanisms**: Add FlashAttention, Linear, etc.
1. **Add Model Architectures**: Implement full transformer models
1. **Profiling Dashboard**: Create visualization tools
1. **Distributed Training**: Extend to multi-GPU setups
1. **Benchmarking Suite**: Automated performance comparison

## 🤝 Contributing

1. Follow the existing code structure
1. Add comprehensive tests for new features
1. Update documentation and type hints
1. Ensure code passes linting and type checking

## 📄 License

[Add your license here]

______________________________________________________________________

**Happy researching!** 🔬✨
