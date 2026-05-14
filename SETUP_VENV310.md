# Setting up Python 3.10 venv for Multi-Seed Training

## Status

**Installation in progress** for `.venv310` (Python 3.10.20).

## What's Needed

### Core Dependencies
```
torch==2.11.0 (with CUDA 11.8)
numpy==2.4.4
scipy==1.17.1
pettingzoo==1.26.1
```

### Installation Command
```bash
cd /home/abz/workspace/PycharmProjects/dif_driven_maddpg
source .venv310/bin/activate

# Install PyTorch with CUDA 11.8 support (for Python 3.10)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy pettingzoo
```

## Verification

After installation:
```bash
source .venv310/bin/activate
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}')"
```

## Why Python 3.10?

- Matches the old Z7S environment
- Good middle ground between compatibility and features
- Test reproducibility across Python versions

## Two Venv Setup

| venv | Python | torch | Status | Use Case |
|---|---|---|---|---|
| `.venvLin` | 3.12.3 | 2.11.0+cu130 | ✅ Ready | Main development, current runs |
| `.venv310` | 3.10.20 | (installing) | 🔄 Setting up | Multi-seed testing, Z7S comparison |

## Running Training

### With .venvLin (default, fast)
```bash
source .venvLin/bin/activate
python run/run_seeds.py --seeds 0 13 9832 --n 4 5 6 --mode full
```

### With .venv310 (comparison, slower)
```bash
source .venv310/bin/activate
python run/run_seeds.py --seeds 100 101 102 --n 5 --mode full
```

## run_seeds.py Script

New configurable runner that replaces the heavy `run_all.py` for pure training:

### Basic Usage
```bash
# Run with custom seeds (test with new seeds 100-102, default folder)
python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full ablation --parallel 3

# Custom output folder (e.g., /tmp for quick test)
python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full ablation \
  --out_root /tmp/test_artifacts --parallel 3

# Quick test: 3 agents, 500 episodes
python run/run_seeds.py --seeds 9832 --n 5 --mode full --episodes 500 --parallel 1

# Dry run to see what would execute
python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full --dry_run

# Force rerun (skip skip-if-done checks)
python run/run_seeds.py --seeds 9832 --n 4 --mode full --rerun
```

### All Options
```
--seeds INT [INT ...]           Seeds to run. Default: [9832, 0, 13]
                                Examples: --seeds 100 101 102, --seeds 42 100 200

--n {4,5,6} [{4,5,6} ...]       Team sizes. Default: [4, 5, 6]
                                Examples: --n 4, --n 4 5 6

--mode {full,ablation,nocoll}   Training modes. Default: [full, ablation]
                                Examples: --mode full, --mode full ablation nocoll

--out_root PATH                 Base output folder. Default: ~/Desktop/dif_driven_revision_artifacts
                                Examples: --out_root /tmp/quick_test
                                         --out_root ./my_artifacts
                                         --out_root /mnt/external_ssd/runs

--episodes INT                  Episodes per run. Default: 1000
                                Examples: --episodes 100, --episodes 500

--max_steps INT                 Max steps per episode. Default: 500

--parallel INT                  Max workers. Default: 3 (use 1 for laptop, 5 for batch)
                                Examples: --parallel 1, --parallel 5

--dry_run                       Print tasks without executing them

--rerun                         Force rerun (ignore completed runs)
```

## Monitoring

Each run generates:
- `{out_root}/n{N}_{mode}_seed{S}/train.log` — subprocess stdout/stderr
- `{out_root}/n{N}_{mode}_seed{S}/meta.json` — timing, seed, episode count, peak GPU
- `{out_root}/n{N}_{mode}_seed{S}/result{N}[_suffix].csv` — per-episode metrics

## Next Steps

1. Wait for `.venv310` torch installation to complete
2. Verify: `source .venv310/bin/activate && python -c "import torch; print(torch.__version__)"`
3. Run dry test: `python run/run_seeds.py --seeds 100 101 102 --n 5 --mode full --episodes 100 --dry_run`
4. Launch: `python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full ablation --parallel 3`

