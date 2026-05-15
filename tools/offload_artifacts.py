"""Copy revision artifacts to external storage and prune local run outputs.

The script is intentionally conservative:

* running training directories with a live ``.run_all.lock`` are skipped;
* files are copied before anything is removed;
* copied files are verified by byte size;
* incomplete runs keep the exact resume files used by ``train_loop``;
* completed runs keep only local scheduler/evaluation essentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path.home() / "Desktop" / "dif_driven_revision_corrected_artifacts"
DEFAULT_TARGET_ROOT = Path("/media/abz/Z7S/experiments_revision_corrected")

RUN_RE = re.compile(r"^n(?P<n>[456])_(?P<mode>full|ablation|nocoll)_seed(?P<seed>\d+)$")
RESULT_SUFFIX = {"full": "", "ablation": "_ablation", "nocoll": "_nocoll"}

COMPLETED_KEEP_NAMES = {
    "meta.json",
    "episode_log.txt",
    "shared_actor.pth",
    "shared_critic.pth",
    "shared_actor_target.pth",
    "shared_critic_target.pth",
}
INCOMPLETE_KEEP_NAMES = COMPLETED_KEEP_NAMES | {
    "training_state.pkl",
    "replay_buffer.pkl",
    "rewards.csv",
    ".run_all.lock",
}
KEEP_SUFFIXES = {".json"}  # Keep offload manifests and small summaries.


@dataclass
class OffloadResult:
    source: Path
    target: Path
    status: str
    copied_files: int = 0
    removed_files: int = 0
    freed_bytes: int = 0
    message: str = ""


@dataclass
class Progress:
    label: str
    total_files: int
    total_bytes: int
    enabled: bool = True
    interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        self.done_files = 0
        self.done_bytes = 0
        self.start_time = time.time()
        self.last_emit = 0.0

    def update(self, *, files: int = 0, bytes_count: int = 0, force: bool = False) -> None:
        self.done_files += files
        self.done_bytes += bytes_count
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last_emit < self.interval_seconds:
            return
        self.last_emit = now

        file_pct = 100.0 if self.total_files == 0 else self.done_files * 100.0 / self.total_files
        byte_pct = 100.0 if self.total_bytes == 0 else self.done_bytes * 100.0 / self.total_bytes
        elapsed = max(now - self.start_time, 1e-6)
        rate_mb_s = self.done_bytes / 1e6 / elapsed
        line = (
            f"\r{self.label}: {self.done_files}/{self.total_files} files "
            f"({file_pct:5.1f}%), {self.done_bytes/1e6:.1f}/{self.total_bytes/1e6:.1f} MB "
            f"({byte_pct:5.1f}%), {rate_mb_s:.1f} MB/s"
        )
        print(line, end="", flush=True)

    def finish(self) -> None:
        if not self.enabled:
            return
        self.update(force=True)
        print("", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy revision artifacts to external storage and prune local files "
            "that are not needed for resume/eval/report scheduling."
        )
    )
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target_root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument(
        "--run_dir",
        type=Path,
        action="append",
        default=[],
        help="Specific run directory to process. May be passed multiple times.",
    )
    parser.add_argument(
        "--all_runs",
        action="store_true",
        help="Process every n{4,5,6}_{mode}_seed{seed} directory under runs/.",
    )
    parser.add_argument(
        "--copy_support_dirs",
        action="store_true",
        help="Also copy source_root/logs and source_root/res without pruning them.",
    )
    parser.add_argument(
        "--keep_local_result_csv",
        action="store_true",
        help="Keep completed run result CSVs locally. Default removes them after verified copy.",
    )
    parser.add_argument(
        "--prune_incomplete_snapshots",
        action="store_true",
        help=(
            "For incomplete runs, remove historical episode_* and replay_buffer_* "
            "snapshots after copy. Resume files training_state.pkl/replay_buffer.pkl "
            "are kept."
        ),
    )
    parser.add_argument(
        "--include_running",
        action="store_true",
        help=(
            "Also process directories with a live .run_all.lock. This only prunes "
            "files not needed for resume; active resume files are kept local."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show intended copy/prune actions without writing or deleting.",
    )
    parser.add_argument(
        "--quiet_progress",
        action="store_true",
        help="Disable progress lines and print only timestamped summary logs.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{stamp}] {message}", flush=True)


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def live_lock(run_dir: Path) -> tuple[bool, int | None]:
    lock_path = run_dir / ".run_all.lock"
    if not lock_path.exists():
        return False, None
    try:
        payload = json.loads(lock_path.read_text())
        pid = int(payload.get("pid") or payload.get("owner_pid") or -1)
    except Exception:
        pid = -1
    return pid_is_running(pid), pid


def ensure_target_root(target_root: Path, *, dry_run: bool = False) -> Path | None:
    target_root = target_root.expanduser()
    if target_root.exists():
        return target_root.resolve()
    if target_root.parent.exists():
        if not dry_run:
            target_root.mkdir(parents=True, exist_ok=True)
        return target_root.resolve()
    return None


def run_result_name(run_dir: Path) -> str | None:
    match = RUN_RE.match(run_dir.name)
    if not match:
        return None
    n = int(match.group("n"))
    mode = match.group("mode")
    return f"result{n}{RESULT_SUFFIX[mode]}.csv"


def run_is_finished(run_dir: Path, target_run_dir: Path | None = None) -> bool:
    meta_path = run_dir / "meta.json"
    meta_finished = False
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta_finished = bool(meta.get("finished")) or meta.get("launcher_status") == "finished"
        except Exception:
            meta_finished = False

    result_name = run_result_name(run_dir)
    local_result = result_name is not None and (run_dir / result_name).exists()
    target_result = (
        result_name is not None
        and target_run_dir is not None
        and (target_run_dir / result_name).exists()
    )
    return bool(meta_finished and (local_result or target_result))


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def total_size(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def copy_tree_verified(
    source: Path,
    target: Path,
    *,
    dry_run: bool,
    progress: bool,
) -> int:
    files = iter_files(source)
    progress_bar = Progress(
        label=f"copy {source.name}",
        total_files=len(files),
        total_bytes=total_size(files),
        enabled=progress and not dry_run,
    )
    copied = 0
    for src_file in files:
        rel = src_file.relative_to(source)
        dst_file = target / rel
        try:
            src_size = src_file.stat().st_size
        except FileNotFoundError:
            progress_bar.update(files=1)
            continue
        copied += 1
        if dry_run:
            log(f"DRY copy {src_file} -> {dst_file}")
            progress_bar.update(files=1, bytes_count=src_size)
            continue

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if not dst_file.exists() or dst_file.stat().st_size != src_size:
            shutil.copy2(src_file, dst_file)

        if dst_file.stat().st_size != src_size:
            raise IOError(f"copy verification failed: {src_file} -> {dst_file}")
        progress_bar.update(files=1, bytes_count=src_size)
    progress_bar.finish()
    return copied


def should_keep_completed(path: Path, run_dir: Path, *, keep_local_result_csv: bool) -> bool:
    rel = path.relative_to(run_dir)
    if len(rel.parts) != 1:
        return False
    name = path.name
    if name in COMPLETED_KEEP_NAMES:
        return True
    if name == "offload_manifest.json":
        return True
    if keep_local_result_csv and name == run_result_name(run_dir):
        return True
    return False


def should_keep_incomplete(path: Path, run_dir: Path, *, prune_snapshots: bool) -> bool:
    rel = path.relative_to(run_dir)
    if len(rel.parts) != 1:
        return False
    name = path.name
    if name in INCOMPLETE_KEEP_NAMES:
        return True
    if name == "offload_manifest.json":
        return True
    if path.suffix in KEEP_SUFFIXES:
        return True
    if not prune_snapshots:
        return True
    return False


def file_is_stable(path: Path, *, settle_seconds: float = 0.2) -> bool:
    try:
        before = path.stat()
        time.sleep(settle_seconds)
        after = path.stat()
    except FileNotFoundError:
        return False
    return (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def prune_run_dir(
    run_dir: Path,
    *,
    completed: bool,
    keep_local_result_csv: bool,
    prune_incomplete_snapshots: bool,
    dry_run: bool,
    progress: bool,
) -> tuple[int, int]:
    removed = 0
    freed = 0
    files = iter_files(run_dir)
    candidate_files = []
    candidate_bytes = 0
    for path in files:
        if completed:
            keep = should_keep_completed(
                path, run_dir, keep_local_result_csv=keep_local_result_csv
            )
        else:
            keep = should_keep_incomplete(
                path, run_dir, prune_snapshots=prune_incomplete_snapshots
            )
        if keep:
            continue
        candidate_files.append(path)
        try:
            candidate_bytes += path.stat().st_size
        except FileNotFoundError:
            pass

    progress_bar = Progress(
        label=f"prune {run_dir.name}",
        total_files=len(candidate_files),
        total_bytes=candidate_bytes,
        enabled=progress,
    )
    for path in candidate_files:
        if not file_is_stable(path):
            log(f"skip changing file {path}")
            progress_bar.update(files=1)
            continue

        try:
            size = path.stat().st_size
        except FileNotFoundError:
            progress_bar.update(files=1)
            continue
        removed += 1
        freed += size
        if dry_run:
            log(f"DRY remove {path} ({size} bytes)")
        else:
            path.unlink()
        progress_bar.update(files=1, bytes_count=size)
    progress_bar.finish()

    if not dry_run:
        for directory in sorted(
            (p for p in run_dir.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
    return removed, freed


def write_manifest(
    run_dir: Path,
    target_run_dir: Path,
    *,
    completed: bool,
    copied_files: int,
    removed_files: int,
    freed_bytes: int,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    manifest = {
        "source": str(run_dir),
        "target": str(target_run_dir),
        "completed": completed,
        "copied_files": copied_files,
        "removed_files": removed_files,
        "freed_bytes": freed_bytes,
        "offloaded_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "offload_manifest.json").write_text(json.dumps(manifest, indent=2))


def offload_run_dir(
    run_dir: Path,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    target_root: Path = DEFAULT_TARGET_ROOT,
    keep_local_result_csv: bool = False,
    prune_incomplete_snapshots: bool = False,
    include_running: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> OffloadResult:
    run_dir = run_dir.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    available_target_root = ensure_target_root(target_root, dry_run=dry_run)
    if available_target_root is None:
        return OffloadResult(
            source=run_dir,
            target=target_root,
            status="target_missing",
            message=f"target root is not available: {target_root}",
        )
    target_root = available_target_root

    if not run_dir.exists():
        return OffloadResult(
            source=run_dir,
            target=target_root,
            status="source_missing",
            message=f"source run dir is missing: {run_dir}",
        )
    if RUN_RE.match(run_dir.name) is None:
        return OffloadResult(
            source=run_dir,
            target=target_root,
            status="skipped_not_run_dir",
            message=f"not a canonical run dir: {run_dir}",
        )

    locked, pid = live_lock(run_dir)
    if locked and not include_running:
        return OffloadResult(
            source=run_dir,
            target=target_root,
            status="skipped_running",
            message=f"live run_all lock pid={pid}: {run_dir}",
        )

    try:
        rel = run_dir.relative_to(source_root)
    except ValueError:
        rel = Path("runs") / run_dir.name
    target_run_dir = target_root / rel

    log(f"copy start: {run_dir} -> {target_run_dir}")
    copied = copy_tree_verified(run_dir, target_run_dir, dry_run=dry_run, progress=progress)
    completed = False if locked else run_is_finished(run_dir, target_run_dir)
    log(f"prune start: {run_dir} completed={completed} locked={locked}")
    removed, freed = prune_run_dir(
        run_dir,
        completed=completed,
        keep_local_result_csv=keep_local_result_csv,
        prune_incomplete_snapshots=prune_incomplete_snapshots,
        dry_run=dry_run,
        progress=progress,
    )
    write_manifest(
        run_dir,
        target_run_dir,
        completed=completed,
        copied_files=copied,
        removed_files=removed,
        freed_bytes=freed,
        dry_run=dry_run,
    )

    return OffloadResult(
        source=run_dir,
        target=target_run_dir,
        status=(
            "offloaded_running"
            if locked
            else "offloaded_completed"
            if completed
            else "offloaded_incomplete"
        ),
        copied_files=copied,
        removed_files=removed,
        freed_bytes=freed,
    )


def copy_support_dir(
    source_root: Path,
    target_root: Path,
    name: str,
    *,
    dry_run: bool,
    progress: bool,
) -> None:
    source = source_root / name
    if not source.exists():
        return
    target = target_root / name
    copied = copy_tree_verified(source, target, dry_run=dry_run, progress=progress)
    log(f"support copied {name}: files={copied}")


def discover_run_dirs(source_root: Path) -> list[Path]:
    runs_dir = source_root / "runs"
    if not runs_dir.exists():
        return []
    return sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and RUN_RE.match(path.name)
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    available_target_root = ensure_target_root(args.target_root, dry_run=args.dry_run)
    if available_target_root is None:
        raise SystemExit(f"Target root is not available: {args.target_root.expanduser()}")
    target_root = available_target_root
    progress = not args.quiet_progress

    if args.copy_support_dirs:
        copy_support_dir(
            source_root,
            target_root,
            "logs",
            dry_run=args.dry_run,
            progress=progress,
        )
        copy_support_dir(
            source_root,
            target_root,
            "res",
            dry_run=args.dry_run,
            progress=progress,
        )

    run_dirs = [path.expanduser() for path in args.run_dir]
    if args.all_runs or not run_dirs:
        run_dirs.extend(discover_run_dirs(source_root))

    seen = set()
    unique_run_dirs = []
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_run_dirs.append(resolved)

    totals = {"copied_files": 0, "removed_files": 0, "freed_bytes": 0}
    statuses: dict[str, int] = {}
    for run_dir in unique_run_dirs:
        result = offload_run_dir(
            run_dir,
            source_root=source_root,
            target_root=target_root,
            keep_local_result_csv=args.keep_local_result_csv,
            prune_incomplete_snapshots=args.prune_incomplete_snapshots,
            include_running=args.include_running,
            dry_run=args.dry_run,
            progress=progress,
        )
        statuses[result.status] = statuses.get(result.status, 0) + 1
        totals["copied_files"] += result.copied_files
        totals["removed_files"] += result.removed_files
        totals["freed_bytes"] += result.freed_bytes
        detail = f" copied={result.copied_files} removed={result.removed_files} freed={result.freed_bytes}"
        if result.message:
            detail += f" message={result.message}"
        log(f"{result.status}: {result.source} -> {result.target}{detail}")

    log(f"summary statuses={statuses} totals={totals}")


if __name__ == "__main__":
    main()
