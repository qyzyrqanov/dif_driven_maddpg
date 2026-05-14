# Revision Experiments - CORRECTED v_ang_max

## Critical Fix Applied

**v_ang_max corrected from π/2 → π/12**

The original Z7S experiments used `v_ang_max=torch.pi/12`, but the revision scripts incorrectly had `torch.pi/2`. This has been fixed in all training scripts.

## Output Location

All results now save to:
```
/media/abz/Z7S/experiments_revision_corrected/
```

This folder will be created automatically. Do NOT use Desktop anymore.

## Running the Experiments

### 1. Full Training Run (all 22 training runs)

```bash
cd /home/abz/workspace/PycharmProjects/dif_driven_maddpg
source .venvLin/bin/activate
export PYTHONPATH=$(pwd)

python run/run_all.py --parallel 5
```

**What this does:**
- Trains all combinations: n∈{4,5,6}, mode∈{full, ablation, nocoll}, seed∈{9832, 0, 13}
- Runs up to 5 in parallel
- Saves to `/media/abz/Z7S/experiments_revision_corrected/runs/`
- Takes ~40-60 GPU hours total

### 2. Quick Verification (2 episodes)

```bash
# n=4, full method, seed 9832
python run/train_done4.py
```

Runs in current directory. Output: `result4.csv`, `result4_ablation.csv`, checkpoints.

## Cleanup

### Should you delete Desktop artifacts?

**YES**, after verifying the media runs are complete:

```bash
rm -rf ~/Desktop/dif_driven_revision_artifacts
```

The old Desktop folder had the incorrect v_ang_max=π/2 setup. The media folder will have the correct π/12.

## Expected Results

With v_ang_max=π/12 (original Z7S setup), episode behavior should match Z7S logs more closely:
- Episode 0: ~-3000 to -3500 mean score (vs -3344.23 in Z7S)
- Episode 1: ~-1000 to -1500 mean score (vs -1233.71 in Z7S)

## Monitoring

Check completeness:
```bash
python run/run_all.py --check_completeness --skip_probe
```

After all training completes:
```bash
python tools/aggregate_revision_results.py
```

This generates final tables and figures in `/media/abz/Z7S/experiments_revision_corrected/res/`

## Parameters Summary

| Property | Value |
|----------|-------|
| Algorithm | IDDPGWithoutS |
| Env | DiffDriveParallelEnvDone |
| **v_ang_max** | **π/12** ✓ (CORRECTED) |
| env_size | 20 |
| num_obstacles | 0 |
| Episode horizon | 500 steps |
| Episodes per run | 1000 |
| Batch size | 128 |
| Replay buffer | 50,000 |
| Seeds | 9832, 0, 13 |
| Modes | full, ablation, nocoll |
