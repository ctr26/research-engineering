# Research Engineering Practices

**Compiled**: 2026-02-16  
**Philosophy**: Minimum viable engineering - just enough structure to enable reusability without slowing down research

---

## Core Principles

1. **Code is communication** - Research code is read by future you, collaborators, and paper reviewers
2. **Reusability over polish** - Make it reusable, not perfect
3. **Fail fast, fail clearly** - Catch errors early with validation, not late with debugging
4. **Document the why, not the what** - Code shows what you did; docs explain why
5. **Automate the boring** - Free up mental energy for research insights

---

## Minimum Viable Engineering (MVE)

### Tier 1: Non-Negotiable (Do Always)
These practices have low cost and high value. No excuses.

#### 1. Version Control
```bash
# Every project starts with:
git init
echo "data/" > .gitignore
echo "models/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
git add .
git commit -m "Initial commit"
```

**Why**: Time travel for code. Answers "what changed?" when experiments break.

**Rules**:
- Commit before each experiment
- Write commit messages that explain the hypothesis: `"Test dropout=0.3 to reduce overfitting"`
- Never commit large data files (use Git LFS or external storage)

#### 2. Project Structure
```
project-name/
├── README.md              # What, why, how to run
├── requirements.txt       # or pyproject.toml
├── data/                  # .gitignored, document source
│   ├── raw/              # Never modify raw data
│   └── processed/        # Derived data, can regenerate
├── notebooks/            # Exploration, numbered: 01_explore.ipynb
├── src/                  # Reusable code
│   ├── __init__.py
│   ├── data.py           # Data loading, processing
│   ├── models.py         # Model definitions
│   └── utils.py          # Helper functions
├── experiments/          # Experiment configs & scripts
│   └── 2026-02-16_baseline.yaml
├── outputs/              # Experiment outputs
│   └── 2026-02-16_baseline/
│       ├── model.pt
│       ├── metrics.json
│       └── logs/
└── tests/                # At minimum: test data processing
    └── test_data.py
```

**Why**: Predictable layout reduces cognitive load. Anyone can navigate.

#### 3. Environment Specification
```toml
# pyproject.toml (recommended) or requirements.txt
[project]
name = "my-research-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "torch>=2.0",
]
```

**Why**: "Works on my machine" → "Works on anyone's machine"

**Python note**: Modern Python (3.10+) is fine. Use `match` statements, `|` for type unions, all the good stuff.

#### 4. Schema Validation
```python
from pydantic import BaseModel, Field, validator
from pathlib import Path

class ExperimentConfig(BaseModel):
    """Configuration for training run."""
    model_name: str = Field(..., description="Model architecture")
    learning_rate: float = Field(1e-3, gt=0, lt=1)
    batch_size: int = Field(32, ge=1)
    data_path: Path
    seed: int = 42
    
    @validator('data_path')
    def path_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"Data path {v} does not exist")
        return v

# Usage
config = ExperimentConfig(
    model_name="resnet50",
    learning_rate=0.001,
    batch_size=32,
    data_path=Path("data/processed/train.pkl")
)
```

**Why**: Catch configuration errors before hours of training. Self-documenting parameters.

#### 5. Type Hints for Public Functions
```python
from typing import Optional
import numpy as np
import pandas as pd

def load_dataset(
    path: Path,
    normalize: bool = True,
    max_samples: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load and optionally preprocess dataset.
    
    Args:
        path: Path to dataset file (.pkl or .csv)
        normalize: If True, scale features to [0, 1]
        max_samples: If set, subsample dataset
        
    Returns:
        X: Features, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
    """
    df = pd.read_csv(path)
    # ... processing ...
    return X, y
```

**Why**: 
- IDE autocomplete and inline docs
- Catch type errors with `mypy` before runtime
- Serves as inline documentation

**Rule**: Type hint all function signatures in `src/`. Optional in notebooks.

#### 6. Basic Unit Tests
```python
# tests/test_data.py
import pytest
from src.data import load_dataset, preprocess

def test_load_dataset_shape():
    """Verify dataset has expected shape."""
    X, y = load_dataset("data/raw/sample.csv")
    assert X.shape[1] == 10, "Expected 10 features"
    assert len(X) == len(y), "X and y must have same length"

def test_normalize_range():
    """Verify normalization produces [0, 1] range."""
    X, _ = load_dataset("data/raw/sample.csv", normalize=True)
    assert X.min() >= 0 and X.max() <= 1, "Normalized data must be in [0, 1]"

def test_missing_file_raises():
    """Verify we fail gracefully on missing data."""
    with pytest.raises(FileNotFoundError):
        load_dataset("data/does_not_exist.csv")
```

**Why**: Data processing bugs corrupt experiments. Tests catch them early.

**What to test**:
- ✅ Data loading and shapes
- ✅ Preprocessing functions (normalization, tokenization)
- ✅ Utility functions (metrics, logging)
- ❌ Model forward passes (too slow, too brittle)
- ❌ End-to-end training (that's an experiment, not a test)

**Run tests**:
```bash
# Install once
pip install pytest

# Run tests
pytest tests/

# Run tests on every commit (optional)
# .git/hooks/pre-commit:
#!/bin/bash
pytest tests/ || exit 1
```

---

### Tier 2: Strongly Recommended (Do for Shared Code)
Higher cost, but essential when multiple people use your code.

#### 7. Docstrings for Public APIs
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = "cuda"
) -> dict[str, list[float]]:
    """Train a PyTorch model.
    
    Args:
        model: Neural network to train (will be modified in-place)
        train_loader: DataLoader with training batches
        optimizer: Optimizer instance (e.g., Adam)
        epochs: Number of full passes through training data
        device: Device to train on ('cuda', 'cpu', or 'mps')
        
    Returns:
        Dictionary with training history:
            - 'loss': List of per-epoch training losses
            - 'accuracy': List of per-epoch training accuracies
            
    Example:
        >>> model = ResNet50(num_classes=10)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> history = train_model(model, train_loader, optimizer, epochs=10)
        >>> print(history['loss'])
        [2.3, 1.8, 1.5, ...]
    """
    # ... implementation ...
```

**Format**: Use Google or NumPy style. Be consistent.

#### 8. Logging Instead of Print
```python
import logging

# Setup (do once per script/module)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/experiment.log"),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Usage
logger.info(f"Starting training with lr={config.learning_rate}")
logger.debug(f"Batch {i}: loss={loss.item():.4f}")  # Only shown if level=DEBUG
logger.warning("Learning rate seems high, may diverge")
logger.error("Training diverged - loss is NaN")
```

**Why**: 
- Persistent logs for long experiments
- Controllable verbosity (DEBUG vs INFO vs WARNING)
- Timestamped, structured output

#### 9. Configuration Files
```yaml
# experiments/2026-02-16_baseline.yaml
experiment:
  name: "baseline_resnet50"
  seed: 42
  device: "cuda"

data:
  train_path: "data/processed/train.pkl"
  val_path: "data/processed/val.pkl"
  batch_size: 32
  num_workers: 4

model:
  architecture: "resnet50"
  pretrained: true
  num_classes: 10

training:
  epochs: 50
  learning_rate: 1e-3
  optimizer: "adam"
  weight_decay: 1e-4
  scheduler: "cosine"

output:
  save_dir: "outputs/2026-02-16_baseline"
  checkpoint_every: 10
```

**Load with Pydantic**:
```python
import yaml
from pydantic import BaseModel

class ExperimentConfig(BaseModel):
    # ... fields matching YAML structure ...
    class Config:
        extra = "forbid"  # Fail on unexpected keys

with open("experiments/config.yaml") as f:
    config_dict = yaml.safe_load(f)
config = ExperimentConfig(**config_dict)
```

**Why**:
- Experiment is self-documenting
- Easy to reproduce
- Can diff configs to see what changed between runs
- Version control configs, not code, to try variations

#### 10. Experiment Tracking
```python
# Minimal: Save config + results
import json
from pathlib import Path

output_dir = Path("outputs/2026-02-16_baseline")
output_dir.mkdir(parents=True, exist_ok=True)

# Save config
with open(output_dir / "config.json", "w") as f:
    json.dump(config.dict(), f, indent=2)

# Save results
results = {
    "final_train_loss": 0.23,
    "final_val_loss": 0.45,
    "final_val_accuracy": 0.89,
    "total_epochs": 50,
    "training_time_minutes": 120
}
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
```

**Alternative**: Use W&B, MLflow, or TensorBoard if team is already on it. But plain JSON is fine for small teams.

---

### Tier 3: Nice to Have (Do for Published Tools)
Only needed if you're releasing code for external use.

#### 11. Continuous Integration
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e .[dev]
      - run: pytest tests/
      - run: mypy src/
      - run: ruff check src/
```

**Why**: Automated quality checks on every commit. Especially useful for collaborative repos.

#### 12. Semantic Versioning
If publishing a library or tool:
- `0.1.0`: Initial release
- `0.1.1`: Bug fix (backwards compatible)
- `0.2.0`: New feature (backwards compatible)
- `1.0.0`: Stable API, backwards compatibility guarantees

**For research projects**: Just use dates or git SHAs. Don't overthink it.

#### 13. Packaging
```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "my-research-tool"
version = "0.1.0"
description = "Tool for analyzing X"
authors = [{name = "Your Name", email = "you@example.com"}]
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "pandas>=2.0"]

[project.optional-dependencies]
dev = ["pytest", "mypy", "ruff"]

[project.scripts]
mytool = "my_tool.cli:main"
```

**Install in dev mode**:
```bash
pip install -e .
```

Now `import my_tool` works anywhere, and `mytool` CLI is available.

---

## Python-Specific Best Practices

### Modern Python (3.10+)

#### Use Built-in Type Unions
```python
# Old (3.9 and earlier)
from typing import Union, Optional
def process(x: Union[int, float]) -> Optional[str]:
    ...

# New (3.10+)
def process(x: int | float) -> str | None:
    ...
```

#### Use Pattern Matching
```python
match result:
    case {"status": "success", "data": data}:
        return data
    case {"status": "error", "message": msg}:
        logger.error(f"Error: {msg}")
        raise ValueError(msg)
    case _:
        raise ValueError("Unexpected result format")
```

#### Use Structural Pattern Matching for Config
```python
match config.model_type:
    case "resnet":
        model = ResNet(config.depth)
    case "vit":
        model = VisionTransformer(config.patch_size)
    case _:
        raise ValueError(f"Unknown model type: {config.model_type}")
```

### Dataclasses for Simple Structures
```python
from dataclasses import dataclass

@dataclass
class TrainingRun:
    experiment_id: str
    timestamp: str
    config: dict
    metrics: dict
    artifacts_path: Path
```

Pydantic is better for validation, dataclasses for simple containers.

### Pathlib Over String Paths
```python
from pathlib import Path

# Good
data_dir = Path("data/raw")
for file in data_dir.glob("*.csv"):
    df = pd.read_csv(file)
    output = data_dir.parent / "processed" / f"{file.stem}_clean.csv"
    df.to_csv(output)

# Avoid
import os
data_dir = "data/raw"
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        # ... messy string manipulation ...
```

### Context Managers for Resources
```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Context manager to time code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{name} took {elapsed:.2f}s")

# Usage
with timer("Data loading"):
    X, y = load_dataset("data/train.csv")

with timer("Training"):
    model.fit(X, y)
```

---

## Data Management Practices

### 1. Never Modify Raw Data
```python
# Bad
df = pd.read_csv("data/experiment_data.csv")
df.dropna(inplace=True)
df.to_csv("data/experiment_data.csv")  # Overwrote raw data!

# Good
raw_df = pd.read_csv("data/raw/experiment_data.csv")
clean_df = raw_df.dropna()
clean_df.to_csv("data/processed/experiment_data_clean.csv")
```

**Why**: Raw data is your source of truth. If preprocessing is buggy, you can rerun.

### 2. Document Data Provenance
```markdown
# data/README.md

## Raw Data

### experiment_data.csv
- **Source**: Scraped from X on 2026-01-15
- **Script**: `scripts/scrape_data.py`
- **Size**: 10,000 samples
- **Columns**: id, timestamp, feature_1, feature_2, label

## Processed Data

### experiment_data_clean.csv
- **Source**: `data/raw/experiment_data.csv`
- **Processing**: `notebooks/01_clean_data.ipynb`
- **Changes**: Removed 234 rows with missing labels, normalized features
- **Date**: 2026-01-16
```

### 3. Version Large Datasets Externally
```bash
# For large files, use DVC, Git LFS, or cloud storage
# Example with DVC:
dvc add data/large_dataset.tar.gz
git add data/large_dataset.tar.gz.dvc .gitignore
git commit -m "Add large dataset (tracked with DVC)"

# Or just document download source:
echo "Download from https://zenodo.org/record/12345" > data/raw/README.md
```

---

## Experiment Management Practices

### 1. Naming Convention
```
outputs/
├── 2026-02-16_baseline_resnet50/
├── 2026-02-17_dropout03/
├── 2026-02-17_augmentation/
└── 2026-02-20_final_model/
```

Format: `YYYY-MM-DD_short_description`

### 2. What to Save
```python
output_dir = Path(f"outputs/{config.experiment_name}")
output_dir.mkdir(parents=True, exist_ok=True)

# Always save:
torch.save(model.state_dict(), output_dir / "model.pt")
with open(output_dir / "config.json", "w") as f:
    json.dump(config.dict(), f, indent=2)
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Optionally save:
# - Training curves (loss, accuracy per epoch)
# - Random seed / git commit hash
# - Hardware info (GPU type, CUDA version)
# - Example predictions (for debugging)
```

### 3. Reproducibility Checklist
```python
import random
import numpy as np
import torch

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full reproducibility (slower):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Call at start of every experiment
set_seed(config.seed)
```

**Also record**:
- Python version
- Library versions (`pip freeze > requirements.txt`)
- Git commit hash: `git rev-parse HEAD`
- GPU type (if using)

---

## Code Review Practices

### When to Request Review

1. **Before submission**: Key experimental code for a paper
2. **Before merging to main**: Any code others will use
3. **When stuck**: "I can't figure out this bug, can you look?"
4. **Optional otherwise**: Research is iterative; not every notebook needs review

### What to Review

- ✅ Correctness: Does the logic match the paper/method?
- ✅ Clarity: Can I understand what this code does?
- ✅ Tests: Are there tests for critical functions?
- ❌ Performance: Unless it's a bottleneck, don't optimize prematurely
- ❌ Style: Auto-formatters (black, ruff format) handle this

### Lightweight Review Process

```markdown
## Pull Request Template

**What**: Brief description of changes

**Why**: What problem does this solve? Related paper/experiment?

**Testing**: 
- [ ] Ran existing tests (`pytest`)
- [ ] Tested manually with X dataset
- [ ] New tests added for new functionality

**Review focus**: 
- Please check correctness of data processing in `src/data.py`
- Ignore notebook mess in `notebooks/exploration/` for now
```

---

## Tooling Recommendations

### Essential
- **Version control**: Git
- **Environment**: Poetry, pip-tools, or conda
- **Testing**: pytest
- **Linting**: ruff (combines isort, flake8, pyupgrade)
- **Formatting**: ruff format or black
- **Type checking**: mypy (run periodically, not necessarily in CI)

### Useful
- **Notebooks**: Jupyter, VSCode notebooks, JupyterLab
- **Experiment tracking**: W&B, MLflow, TensorBoard (or just JSON files)
- **Validation**: Pydantic
- **Documentation**: Sphinx (for libraries), mkdocs (for projects)

### Advanced (Only if Needed)
- **DVC**: Data version control for large datasets
- **Pre-commit hooks**: Auto-run formatters and linters
- **Profiling**: py-spy, line_profiler (when performance matters)
- **Coverage**: pytest-cov (useful for shared libraries)

---

## Anti-Patterns to Avoid

### ❌ Premature Optimization
Don't spend hours optimizing code that runs once. Optimize hot loops, not one-off scripts.

### ❌ Over-Engineering
Research code doesn't need microservices, elaborate architectures, or 100% test coverage.

### ❌ No Version Control
"I'll add Git later" = never. Start with Git from day 1.

### ❌ Hardcoded Paths
```python
# Bad
df = pd.read_csv("/home/yourname/projects/data/train.csv")

# Good
from pathlib import Path
DATA_DIR = Path(__file__).parent / "data"
df = pd.read_csv(DATA_DIR / "train.csv")
```

### ❌ Giant Notebooks
If a notebook is >500 lines, extract functions to `src/`. Notebooks are for exploration and presentation, not production logic.

### ❌ No Documentation
Future you will forget why you made that decision. Write it down.

---

## Migration Path: From Chaotic to Structured

### Phase 1: Survival Mode (Week 1)
- Create project structure
- Add Git
- Write a basic README
- Create requirements.txt

### Phase 2: Minimum Viable (Week 2-3)
- Add type hints to key functions
- Write tests for data processing
- Use config files instead of command-line arguments
- Set up logging

### Phase 3: Team-Ready (Month 2)
- Add docstrings
- Set up CI (GitHub Actions)
- Code review for shared code
- Standardize experiment tracking

### Phase 4: Publication-Ready (As Needed)
- Full test coverage for published code
- Comprehensive documentation
- Release as package if it's a tool others will use

---

## Summary: The MVE Stack

| Practice | When | Cost | Value |
|----------|------|------|-------|
| Git | Always | Low | High |
| Project structure | Always | Low | High |
| Type hints | Public APIs | Low | High |
| Schema validation | Configs | Low | High |
| Unit tests | Data processing | Medium | High |
| Logging | Long experiments | Low | Medium |
| Config files | Multi-run experiments | Low | High |
| Docstrings | Shared code | Medium | Medium |
| CI | Team projects | Medium | Medium |
| Code review | Pre-publication | High | High |

**Bottom line**: Start with Git, structure, types, and tests. Add the rest as needed.
