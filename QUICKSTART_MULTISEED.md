# Quick Start: Multi-Seed Training with Custom Seeds

## ✅ Setup Complete

Both environments are ready:

### .venvLin (Main, Python 3.12)
```
torch: 2.11.0+cu130
CUDA: ✓ Available
Status: ✓ Ready
```

### .venv310 (Comparison, Python 3.10)
```
torch: 2.7.1+cu118
CUDA: ✓ Available
Status: ✓ Ready
```

---

## 1. Run Custom Seeds (Quick Test)

Test with new seeds 100, 101, 102 using .venv310:

```bash
cd /home/abz/workspace/PycharmProjects/dif_driven_maddpg
source .venv310/bin/activate
export PYTHONPATH=$(pwd)

# Dry run first (see what will execute)
python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full --episodes 100 --dry_run

# Actually run (100 episodes per seed, 3 parallel workers, default output folder)
python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full --episodes 100 --parallel 3

# With custom output folder (e.g., /tmp for quick test, or external SSD)
python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full --episodes 100 \
  --out_root /tmp/test_artifacts --parallel 3
```

---

## 2. Compare Python 3.10 vs 3.12

### With Python 3.10 (testing reproducibility)
```bash
source .venv310/bin/activate
python run/run_seeds.py --seeds 200 201 202 --n 5 --mode ablation --episodes 500 --parallel 1
```

### With Python 3.12 (main codebase)
```bash
source .venvLin/bin/activate
python run/run_seeds.py --seeds 200 201 202 --n 5 --mode ablation --episodes 500 --parallel 1
```

Compare outputs in:
- `.../n5_ablation_seed200/meta.json` (timing, GPU memory, episodes)
- `.../n5_ablation_seed200/result5_ablation.csv` (episode metrics)

---

## 3. Full Multi-Seed Batch

Run the recommended 2 new seeds (100, 101) across all configurations:

```bash
source .venv310/bin/activate

# Default output: ~/Desktop/dif_driven_revision_artifacts/
python run/run_seeds.py \
  --seeds 100 101 \
  --n 4 5 6 \
  --mode full ablation nocoll \
  --episodes 1000 \
  --parallel 3

# Or save to custom folder (e.g., external SSD, /tmp, or local directory)
python run/run_seeds.py \
  --seeds 100 101 \
  --n 4 5 6 \
  --mode full ablation nocoll \
  --episodes 1000 \
  --out_root /mnt/external_ssd/dif_driven \
  --parallel 3
```

**Expected**: 18 total training runs (2 seeds × 3 n × 3 modes)
**Time**: ~36-48 GPU-hours (3 parallel workers)
**Default output**: `~/Desktop/dif_driven_revision_artifacts/n{N}_{mode}_seed{S}/`
**Custom output**: Any folder you specify with `--out_root`

---

## 4. Configuration Examples

### Minimal (laptop test, temp folder)
```bash
python run/run_seeds.py --seeds 100 --n 5 --mode full --episodes 100 \
  --out_root /tmp/quick_test --parallel 1
```
→ 1 run, ~10 minutes, no parallelism, outputs to `/tmp/quick_test/`

### Medium (multi-seed, single config, local folder)
```bash
python run/run_seeds.py --seeds 100 101 102 --n 4 --mode full --episodes 500 \
  --out_root ./my_artifacts --parallel 3
```
→ 3 runs, ~3 hours each, 3 parallel workers, outputs to `./my_artifacts/`

### Large (full revision batch, external SSD)
```bash
python run/run_seeds.py --seeds 100 101 --n 4 5 6 --mode full ablation nocoll \
  --episodes 1000 --out_root /mnt/external_ssd/revision --parallel 5
```
→ 18 runs, 1000 episodes, all modes, 5 parallel workers, outputs to `/mnt/external_ssd/revision/`

### Default (Desktop, no --out_root specified)
```bash
python run/run_seeds.py --seeds 100 101 --n 4 5 6 --mode full ablation --episodes 1000 --parallel 3
```
→ Outputs to `~/Desktop/dif_driven_revision_artifacts/` (default)

---

## 5. Understanding Outputs

Each run creates an output directory with:

```
~/Desktop/dif_driven_revision_artifacts/n4_full_seed100/
├── train.log              # Subprocess stdout/stderr
├── meta.json              # Timing, seed, GPU memory, episodes completed
├── result4.csv            # Per-episode metrics (success rate, completion time, return)
├── shared_actor.pth       # Final trained actor network
├── shared_critic.pth      # Final trained critic network
├── replay_buffer.pkl      # Replay buffer state (can be large)
└── *.png                  # Training plots
```

**Key file**: `meta.json` shows:
```json
{
  "n": 4,
  "mode": "full",
  "seed": 100,
  "episodes": 1000,
  "wall_seconds": 7200,
  "peak_gpu_bytes": 2147483648,
  "host": "machine-name",
  "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU"
}
```

---

## 6. Dry Run (No Execution)

Always test first without running:

```bash
python run/run_seeds.py --seeds 100 101 --n 4 5 --mode full --dry_run
```

Output shows exactly which tasks would be created and where they'd write artifacts.

---

## 7. Resume / Rerun

### Resume incomplete runs (default behavior)
```bash
python run/run_seeds.py --seeds 100 101 --n 4 5 --mode full
```
→ Automatically skips tasks with completed `result*.csv` files

### Force rerun (ignore existing results)
```bash
python run/run_seeds.py --seeds 100 101 --n 4 5 --mode full --rerun
```
→ Runs all tasks even if outputs exist

---

## 8. Monitoring Active Runs

While training is in progress:

```bash
# Watch logs (all runs)
tail -f ~/Desktop/dif_driven_revision_artifacts/logs/*.log

# Watch specific run
tail -f ~/Desktop/dif_driven_revision_artifacts/n4_full_seed100/train.log

# GPU usage
nvidia-smi -l 1

# Disk usage
du -sh ~/Desktop/dif_driven_revision_artifacts/
```

---

## 9. Script Comparison: run_all.py vs run_seeds.py

| Feature | run_all.py | run_seeds.py |
|---|---|---|
| Purpose | Full revision orchestration | Training only |
| Includes | Training + eval + probes | Training only |
| Complexity | Heavy (2000+ lines) | Light (200 lines) |
| Flexibility | Fixed seed list [9832, 0, 13] | Arbitrary seeds |
| Config | Hardcoded | Full argparse |
| Output | External artifact root | Configurable |

**When to use**:
- `run_all.py`: Full revision pipeline (training → eval → results)
- `run_seeds.py`: Quick multi-seed tests, custom seeds, flexibility

---

## 10. Next: Analysis

After training completes, analyze results:

```bash
# Per-run metrics
cat ~/Desktop/dif_driven_revision_artifacts/n4_full_seed100/meta.json | python -m json.tool

# Summary across seeds
for seed in 100 101 102; do
  echo "Seed $seed:";
  tail -1 ~/Desktop/dif_driven_revision_artifacts/n4_full_seed$seed/result4.csv;
done
```

---

## Quick Reference

### Basic Commands (default output folder)
```bash
# Test new seeds 100-102, all configs, 100 episodes (to ~/Desktop/...)
python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full ablation nocoll --episodes 100 --parallel 3

# Just the full method, quick
python run/run_seeds.py --seeds 100 101 --n 5 --mode full --episodes 500

# Resume incomplete runs (skips done tasks automatically)
python run/run_seeds.py --seeds 100 101 102 --n 4 5 6 --mode full ablation nocoll
```

### Custom Output Folders
```bash
# Save to /tmp for quick temporary runs
python run/run_seeds.py --seeds 100 --n 5 --mode full --episodes 100 --out_root /tmp/test

# Save to external SSD
python run/run_seeds.py --seeds 100 101 --n 4 5 6 --mode full ablation --episodes 1000 \
  --out_root /mnt/external_ssd/dif_driven --parallel 5

# Save to local relative directory
python run/run_seeds.py --seeds 100 --n 4 --mode full --episodes 500 --out_root ./my_runs

# Compare Python 3.10 vs 3.12 with labeled folders
source .venv310/bin/activate && python run/run_seeds.py --seeds 100 --n 5 --mode full \
  --episodes 100 --out_root ./results_py310 --parallel 1
source .venvLin/bin/activate && python run/run_seeds.py --seeds 100 --n 5 --mode full \
  --episodes 100 --out_root ./results_py312 --parallel 1
```

---

## Questions?

Check logs:
```bash
tail -100 ~/Desktop/dif_driven_revision_artifacts/logs/*.log
```

Or inspect a run's training log:
```bash
cat ~/Desktop/dif_driven_revision_artifacts/n5_ablation_seed100/train.log
```

