#!/usr/bin/env python3
"""Aggregate final multi-seed results for the offline_replay_restart_v3 sweep.

Reads every run directory under <root>/runs/n{N}_{mode}_seed{S}/, computes the
last-W-episode success rate (all-N landmarks covered) and coverage (mean fraction
of landmarks covered), and emits:
  * a per-run CSV  (revision_final_summary.csv)
  * a per-cell mean +/- SD CSV (revision_final_cell_summary.csv)
  * a compact stdout summary (<1 page)

Success rate here is the *training-window* SR (exploration noise on), matching the
metric used throughout the restart-v3 notes. Use --window to change the window.
"""
import argparse, csv, glob, json, os, statistics, sys

MODES = ["full", "ablation", "nocoll"]
NS = [4, 5, 6]


def run_metrics(run_dir, n, window):
    csvs = glob.glob(os.path.join(run_dir, "result*.csv")) or \
           glob.glob(os.path.join(run_dir, "rewards.csv"))
    if not csvs:
        return None
    f = max(csvs, key=os.path.getmtime)
    ep = {}
    with open(f) as fh:
        rd = csv.DictReader(fh)
        if "episode_id" not in (rd.fieldnames or []) or "done_count" not in rd.fieldnames:
            return None
        for row in rd:
            try:
                eid = int(float(row["episode_id"])); dc = float(row["done_count"])
            except (TypeError, ValueError):
                continue
            ep[eid] = max(ep.get(eid, 0.0), dc)
    if not ep:
        return None
    eids = sorted(ep)
    last = eids[-window:]
    sr = 100.0 * sum(1 for e in last if ep[e] >= n) / len(last)
    cov = 100.0 * sum(ep[e] / n for e in last) / len(last)
    return {"sr": sr, "cov": cov, "episodes": len(eids), "csv": os.path.basename(f)}


def meta_of(run_dir):
    try:
        return json.load(open(os.path.join(run_dir, "meta.json")))
    except Exception:
        return {}


def restart_count(run_dir):
    try:
        return int(json.load(open(os.path.join(run_dir, "restart_state.json"))).get("restart_count", 0))
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/abz/Desktop/dif_driven_revision_offline_replay_restart_v3_artifacts")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--out_dir", default=None, help="where to write CSVs (default <root>/res)")
    args = ap.parse_args()

    runs_root = os.path.join(args.root, "runs")
    out_dir = args.out_dir or os.path.join(args.root, "res")
    os.makedirs(out_dir, exist_ok=True)

    per_run = []
    for n in NS:
        for mode in MODES:
            for seed in args.seeds:
                name = f"n{n}_{mode}_seed{seed}"
                d = os.path.join(runs_root, name)
                if not os.path.isdir(d):
                    continue
                m = run_metrics(d, n, args.window)
                meta = meta_of(d)
                row = {
                    "run": name, "n": n, "mode": mode, "seed": seed,
                    "sr": None if m is None else round(m["sr"], 1),
                    "cov": None if m is None else round(m["cov"], 1),
                    "episodes": None if m is None else m["episodes"],
                    "restarts": restart_count(d),
                    "train_hours": round(meta.get("total_train_hours", 0.0), 3) or None,
                    "peak_gpu_gb": round(meta.get("peak_gpu_gb", 0.0), 4) or None,
                    "finished": meta.get("finished"),
                }
                per_run.append(row)

    # per-run CSV
    run_csv = os.path.join(out_dir, "revision_final_summary.csv")
    with open(run_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_run[0].keys()))
        w.writeheader(); w.writerows(per_run)

    # per-cell aggregation
    cell_rows = []
    for n in NS:
        for mode in MODES:
            vals = [r for r in per_run if r["n"] == n and r["mode"] == mode and r["sr"] is not None]
            if not vals:
                continue
            srs = [r["sr"] for r in vals]; covs = [r["cov"] for r in vals]
            cell_rows.append({
                "n": n, "mode": mode, "seeds": len(vals),
                "sr_mean": round(statistics.mean(srs), 1),
                "sr_sd": round(statistics.pstdev(srs), 1) if len(srs) > 1 else 0.0,
                "cov_mean": round(statistics.mean(covs), 1),
                "cov_sd": round(statistics.pstdev(covs), 1) if len(covs) > 1 else 0.0,
                "sr_per_seed": ", ".join(f"{r['seed']}:{r['sr']}" for r in sorted(vals, key=lambda x: x["seed"])),
            })
    cell_csv = os.path.join(out_dir, "revision_final_cell_summary.csv")
    with open(cell_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cell_rows[0].keys()))
        w.writeheader(); w.writerows(cell_rows)

    # stdout summary
    print(f"window={args.window}  runs_found={len(per_run)}")
    print(f"per-run CSV : {run_csv}")
    print(f"per-cell CSV: {cell_csv}\n")
    print(f"{'n':>2} {'mode':<9} {'SR mean+/-SD':>14} {'cov mean+/-SD':>14}  per-seed SR")
    for r in cell_rows:
        print(f"{r['n']:>2} {r['mode']:<9} {r['sr_mean']:>6.1f}+/-{r['sr_sd']:<5.1f} "
              f"{r['cov_mean']:>6.1f}+/-{r['cov_sd']:<5.1f}  [{r['sr_per_seed']}]")
    # by-mode overall
    print()
    for mode in MODES:
        srs = [r["sr"] for r in per_run if r["mode"] == mode and r["sr"] is not None]
        print(f"overall {mode:<9}: {statistics.mean(srs):5.1f} +/- {statistics.pstdev(srs):4.1f}  (k={len(srs)})")
    # compute
    hrs = [r["train_hours"] for r in per_run if r["train_hours"]]
    gpus = [r["peak_gpu_gb"] for r in per_run if r["peak_gpu_gb"]]
    if hrs:
        print(f"\ncompute: train_hours mean {statistics.mean(hrs):.2f} "
              f"(min {min(hrs):.2f}, max {max(hrs):.2f}); total {sum(hrs):.1f} GPU-h")
    if gpus:
        print(f"peak_gpu_gb mean {statistics.mean(gpus):.3f} (max {max(gpus):.3f})")


if __name__ == "__main__":
    main()
