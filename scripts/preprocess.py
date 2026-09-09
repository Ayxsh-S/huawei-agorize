"""Build the processed cache: raw Huawei PLYs -> ``cache/<subject>_<side>.npz`` (Role A).

Run by Alfred against the NDA-protected data; this is what Roles B and C train
and validate on (read it back with ``src.cache.load_cached_ear``)::

    python scripts/preprocess.py --subject-list splits/train_ids.txt --with-targets
    python scripts/preprocess.py --subject-list splits/val_ids.txt   --with-targets
    python scripts/preprocess.py --subject-list splits/test_ids.txt  --workers 4

For every listed subject and side:

1. ``load_mesh`` -> ``canonicalize_ear`` with the FROZEN ``configs/crop.yaml``
   (mesh + config only — the transform never sees a landmark, see the
   no-leakage rule in ``CLAUDE.md``),
2. with ``--with-targets``, the GT landmarks are pushed *through* that
   transform and **verified**: ``inverse_transform_points`` on the canonical
   targets (through the transform as it will be stored and reloaded) must
   reproduce the original GT to < 1e-9 mm, or the run aborts naming the
   subject. Nothing is written for an ear that fails this. The cache is
   therefore self-validating: a target in it is known to map back exactly,
3. ``src.cache.write_cached_ear`` writes the npz atomically.

Sampling seed: each ear draws its 2048 points from a seed derived from
``(--seed, subject_id, side)`` by sha256 (:func:`ear_seed`), so every cache
entry is reproducible on its own and the two ears of a subject never share a
draw. The per-ear seed is stored in the file.

Existing files are skipped unless ``--overwrite`` — but only when they were
built with the same n_points, seed, crop config and target policy; a stale
entry is a failure, not a skip, so two runs with different settings can never
be quietly mixed in one cache.

Prints a progress line every 25 subjects and a final summary (ears written /
skipped / failed, cache size on disk, min / median / max crop vertices).
Aggregates and subject IDs only — never a coordinate. **Exit code 1 on any
failure.**
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
sys.path.insert(0, str(REPO_ROOT))  # for `src`
sys.path.insert(0, str(_HERE))         # sibling scripts, for shared helpers

from crop_stats import describe_crop_config, read_subject_list, sha256_file  # noqa: E402
from src.cache import (  # noqa: E402
    CACHE_SUFFIX,
    SCHEMA_VERSION,
    cache_path,
    list_cached,
    write_cached_ear,
)
from src.data import (  # noqa: E402
    DATA_ROOT,
    SIDES,
    list_subjects,
    load_mesh,
    load_subject_landmarks,
    mesh_path,
)
from src.geometry import (  # noqa: E402
    DEFAULT_CROP_CONFIG,
    N_POINTS,
    CropConfig,
    EarTransform,
    _resolve_config_path,
    canonicalize_ear,
    inverse_transform_points,
    load_crop_config,
    transform_points_to_canonical,
)

# Max |inverse(canonical(GT)) - GT| allowed before an ear's targets are refused.
TARGET_TOL = 1e-9   # mm
PROGRESS_EVERY = 25
MAX_LISTED = 20

STATUS_WRITTEN = "written"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_TARGET_MISMATCH = "target_mismatch"   # fatal: aborts the whole run


class TargetRoundTripError(RuntimeError):
    """The canonical GT did not invert back to the original GT within TARGET_TOL."""


def ear_seed(seed: int, subject_id: str, side: str) -> int:
    """Deterministic, distinct sampling seed for one ear.

    sha256 of ``"<seed>|<subject_id>|<side>"``, first 8 bytes, top bit dropped
    so it fits an int64 npz field. Stable across Python versions and processes
    (``hash()`` is neither), and distinct for the two ears of one subject.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    digest = hashlib.sha256(f"{int(seed)}|{subject_id}|{side}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def canonical_targets(landmarks: np.ndarray, transform: EarTransform) -> tuple[np.ndarray, float]:
    """``([85,3] float64 canonical GT, max round-trip error in mm)``.

    The landmarks are only mapped *through* ``transform`` (which was built
    from the mesh alone) and the result is checked to invert exactly through
    the serialised transform — ``EarTransform.from_dict(to_dict())`` — because
    that is the object Role D will invert predictions with after reloading
    the cache. Raises :class:`TargetRoundTripError` above :data:`TARGET_TOL`.
    """
    gt = np.asarray(landmarks, dtype=np.float64)
    canonical = transform_points_to_canonical(gt, transform)
    cached = EarTransform.from_dict(transform.to_dict())
    restored = inverse_transform_points(canonical, cached)
    error = float(np.max(np.abs(restored - gt)))
    if not error < TARGET_TOL:   # also catches nan
        raise TargetRoundTripError(
            f"inverse_transform_points(targets) is off by {error:.3e} mm "
            f"(tolerance {TARGET_TOL:.0e})"
        )
    return canonical, error


@dataclass(frozen=True)
class Job:
    """Everything one worker needs for one subject (picklable, no arrays of data)."""

    subject_id: str
    root: Path
    out_dir: Path
    configs: dict[str, CropConfig]
    n_points: int
    seed: int
    with_targets: bool
    overwrite: bool
    crop_config_sha256: str
    split_file: str
    split_sha256: str


@dataclass
class EarResult:
    subject_id: str
    side: str
    status: str
    n_crop_vertices: int | None = None
    suspicious: bool = False
    with_replacement: bool = False
    roundtrip_error: float | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        return f"{self.subject_id} {self.side}"


def _stale_reason(path: Path, job: Job) -> str | None:
    """Why an existing cache file cannot stand in for this run, or ``None`` if it can."""
    try:
        with np.load(path, allow_pickle=False) as npz:
            keys = set(npz.files)
            checks = {
                "schema_version": (str(npz["schema_version"]), SCHEMA_VERSION),
                "n_points": (int(npz["n_points"]), int(job.n_points)),
                "seed": (int(npz["seed"]), int(job.seed)),
                "crop_config_sha256": (str(npz["crop_config_sha256"]), job.crop_config_sha256),
            }
            has_targets = "targets" in keys
    except Exception as exc:
        return f"existing file unreadable ({type(exc).__name__}: {exc})"
    for name, (found, wanted) in checks.items():
        if found != wanted:
            return f"existing file has {name}={found!r}, this run wants {wanted!r}"
    if job.with_targets and not has_targets:
        return "existing file has no targets but --with-targets was requested"
    if has_targets and not job.with_targets:
        # Symmetric on purpose: a label-free cache (val/test/inference) must
        # not quietly inherit GT landmarks from an earlier --with-targets run.
        return "existing file carries targets but this run is without --with-targets"
    return None


def process_subject(job: Job) -> list[EarResult]:
    """Cache both ears of one subject. Never raises; failures come back as results."""
    sid = job.subject_id
    results: list[EarResult] = []
    pending: list[str] = []
    for side in SIDES:
        path = cache_path(job.out_dir, sid, side)
        if job.overwrite or not path.exists():
            pending.append(side)
            continue
        reason = _stale_reason(path, job)
        if reason is None:
            results.append(EarResult(sid, side, STATUS_SKIPPED))
        else:
            results.append(EarResult(sid, side, STATUS_FAILED,
                                     error=f"stale cache entry, rerun with --overwrite: {reason}"))
    if not pending:
        return results

    try:
        raw = load_mesh(mesh_path(sid, job.root))
    except Exception as exc:
        msg = f"load failed: {type(exc).__name__}: {exc}"
        return results + [EarResult(sid, side, STATUS_FAILED, error=msg) for side in pending]

    landmarks = None
    if job.with_targets:
        try:
            landmarks = load_subject_landmarks(sid, job.root)
        except Exception as exc:
            msg = f"landmarks failed: {type(exc).__name__}: {exc}"
            return results + [EarResult(sid, side, STATUS_FAILED, error=msg) for side in pending]

    for side in pending:
        try:
            # Mesh + frozen config only. `landmarks` is deliberately not in
            # reach of this call: the transform must never see GT.
            ear = canonicalize_ear(
                raw, side, job.configs[side], n_points=job.n_points,
                seed=ear_seed(job.seed, sid, side),
            )
            if ear.qa["suspicious"]:
                # Same rule as scripts/check_crop_all.py: a crop this small is
                # a data or config problem to look at, not an ear to train on.
                results.append(EarResult(
                    sid, side, STATUS_FAILED, suspicious=True,
                    n_crop_vertices=int(ear.qa["n_vertices"]),
                    error=f"suspicious crop: {ear.qa['n_vertices']} vertices < "
                          f"min_vertices {job.configs[side].min_vertices}; nothing written",
                ))
                continue
            targets = None
            error = None
            if landmarks is not None:
                targets, error = canonical_targets(landmarks[side], ear.transform)
            write_cached_ear(
                cache_path(job.out_dir, sid, side), ear,
                seed=job.seed, ear_seed=ear_seed(job.seed, sid, side),
                crop_config_sha256=job.crop_config_sha256,
                split_file=job.split_file, split_sha256=job.split_sha256,
                targets=targets,
            )
        except TargetRoundTripError as exc:
            # Fatal for the whole run: stop at the first ear, so the abort
            # names the ear that actually failed and nothing after it is built.
            results.append(EarResult(sid, side, STATUS_TARGET_MISMATCH, error=str(exc)))
            break
        except Exception as exc:
            results.append(EarResult(sid, side, STATUS_FAILED,
                                     error=f"{type(exc).__name__}: {exc}"))
            continue
        results.append(EarResult(
            sid, side, STATUS_WRITTEN,
            n_crop_vertices=int(ear.qa["n_vertices"]),
            suspicious=bool(ear.qa["suspicious"]),
            with_replacement=bool(ear.qa["sampled_with_replacement"]),
            roundtrip_error=error,
        ))
    return results


@dataclass
class RunSummary:
    written: list[EarResult] = field(default_factory=list)
    skipped: list[EarResult] = field(default_factory=list)
    failed: list[EarResult] = field(default_factory=list)
    aborted_on: EarResult | None = None

    def add(self, r: EarResult) -> None:
        if r.status == STATUS_WRITTEN:
            self.written.append(r)
        elif r.status == STATUS_SKIPPED:
            self.skipped.append(r)
        elif r.status == STATUS_TARGET_MISMATCH:
            if self.aborted_on is None:      # the first one is the one to name
                self.aborted_on = r
        else:
            self.failed.append(r)

    @property
    def ok(self) -> bool:
        return (self.aborted_on is None and not self.failed
                and (self.written or self.skipped))


def _crop_stats_line(label: str, results: list[EarResult]) -> str:
    sizes = np.asarray([r.n_crop_vertices for r in results], dtype=np.float64)
    if sizes.size == 0:
        return f"  {label:<22}(none written)"
    return (f"  {label:<22}min {sizes.min():.0f}   median {np.median(sizes):.0f}   "
            f"max {sizes.max():.0f}")


def _labels(results: list[EarResult]) -> str:
    shown = ", ".join(r.label for r in results[:MAX_LISTED])
    more = "" if len(results) <= MAX_LISTED else f" (+{len(results) - MAX_LISTED} more)"
    return shown + more


def sweep_temp_files(out_dir: Path) -> list[Path]:
    """Delete leftovers of interrupted writes (``.<name>.npz.<pid>.tmp``).

    A worker killed by ``pool.terminate()`` on an abort can leave one behind;
    ``list_cached`` never lists them, but they should not pile up either.
    """
    removed = []
    if out_dir.is_dir():
        for p in out_dir.glob(f".*{CACHE_SUFFIX}.*.tmp"):
            try:
                p.unlink()
            except OSError:
                # Held open by a concurrent run (Windows refuses the unlink):
                # that run's own os.replace will consume it. Leave it.
                continue
            removed.append(p)
    return removed


def _human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def print_summary(summary: RunSummary, out_dir: Path, with_targets: bool) -> None:
    print("\n=== CACHE ===")
    print(f"  ears written: {len(summary.written)}")
    print(f"  ears skipped (already cached, same settings): {len(summary.skipped)}")
    print(f"  ears failed:  {len(summary.failed)}")
    if summary.failed:
        for r in summary.failed[:MAX_LISTED]:
            print(f"    {r.label}: {r.error}")
        if len(summary.failed) > MAX_LISTED:
            print(f"    ... and {len(summary.failed) - MAX_LISTED} more")

    try:
        files = [p for _, _, p in list_cached(out_dir)]
        total = sum(p.stat().st_size for p in files)
        print(f"  cache on disk: {len(files)} files, {_human_size(total)} in {out_dir}")
    except FileNotFoundError:
        print(f"  cache on disk: nothing under {out_dir}")

    print("  crop vertices (ears written this run):")
    print(_crop_stats_line("all sides", summary.written))
    for side in SIDES:
        print(_crop_stats_line(side, [r for r in summary.written if r.side == side]))
    suspicious = [r for r in summary.failed if r.suspicious]
    print(f"  suspicious crops (below CropConfig.min_vertices; FAILED, not written): "
          f"{len(suspicious)}" + (f"  {_labels(suspicious)}" if suspicious else ""))
    replaced = [r for r in summary.written if r.with_replacement]
    print(f"  ears sampled WITH replacement (crop smaller than n_points): "
          f"{len(replaced)}" + (f"  {_labels(replaced)}" if replaced else ""))
    if replaced:
        print("    !! written, but those ears repeat points; check them with "
              "scripts/plot_ear.py before training on them")
    if with_targets and summary.written:
        errs = [r.roundtrip_error for r in summary.written if r.roundtrip_error is not None]
        if errs:
            print(f"  targets: max |inverse(canonical(GT)) - GT| = {max(errs):.3e} mm "
                  f"over {len(errs)} ears (tolerance {TARGET_TOL:.0e}); every stored "
                  "target inverts exactly")

    if summary.aborted_on is not None:
        r = summary.aborted_on
        print(f"  => ABORTED on {r.label}: {r.error}. Nothing was written for that ear; "
              "the canonical transform is not exactly invertible there.")
    elif summary.failed:
        print("  => FAIL - some ears could not be cached (see above). Fix the cause and "
              "re-run; already-written ears will be skipped.")
    elif not summary.written and not summary.skipped:
        print("  => FAIL - nothing was cached.")
    else:
        print("  => OK - every listed ear is cached with these settings.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=DATA_ROOT,
                    help="dataset root (default: $HUAWEI_DATA_ROOT)")
    ap.add_argument("--subject-list", type=Path, required=True,
                    help="REQUIRED text file of subject IDs, one per line "
                         "(e.g. splits/train_ids.txt)")
    ap.add_argument("--out", type=Path, default=Path("cache"),
                    help="cache directory (default: cache/)")
    ap.add_argument("--crop", type=Path, default=Path(DEFAULT_CROP_CONFIG),
                    help="frozen crop config (default: configs/crop.yaml)")
    ap.add_argument("--n-points", type=int, default=N_POINTS,
                    help=f"points sampled per ear (default {N_POINTS})")
    ap.add_argument("--seed", type=int, default=0,
                    help="run seed; each ear's sampling seed is derived from "
                         "(seed, subject_id, side)")
    ap.add_argument("--with-targets", action="store_true",
                    help="also store the canonical GT landmarks (train/dev only); "
                         "each ear's targets are verified to invert exactly")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild ears that are already cached")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes (default 1)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cache only the first N listed subjects (0 = all)")
    args = ap.parse_args(argv)
    if args.n_points <= 0:
        ap.error("--n-points must be > 0")
    if args.workers <= 0:
        ap.error("--workers must be >= 1")
    return args


def _run(jobs: list[Job], workers: int, summary: RunSummary) -> None:
    """Feed jobs to the workers, printing progress; stops at the first target mismatch."""
    n = len(jobs)

    def consume(i: int, results: list[EarResult]) -> bool:
        for r in results:
            summary.add(r)
        if i % PROGRESS_EVERY == 0 or i == n:
            print(f"  ... {i}/{n} subjects  (written {len(summary.written)}, "
                  f"skipped {len(summary.skipped)}, failed {len(summary.failed)})",
                  flush=True)
        return summary.aborted_on is None

    if workers == 1:
        for i, job in enumerate(jobs, 1):
            if not consume(i, process_subject(job)):
                return
        return

    # spawn everywhere (Windows has nothing else) so behaviour does not depend
    # on the platform; `with` terminates the pool on an early return.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=min(workers, n)) as pool:
        for i, results in enumerate(pool.imap(process_subject, jobs, chunksize=1), 1):
            if not consume(i, results):
                return


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser()
    split_path = Path(args.subject_list).expanduser()
    out_dir = Path(args.out).expanduser()

    if not split_path.is_file():
        print(f"!! subject list not found: {split_path}")
        return 1
    try:
        wanted_ids = read_subject_list(split_path)
    except ValueError as exc:
        print(f"!! {exc}")
        return 1

    try:
        configs = {side: load_crop_config(side, args.crop) for side in SIDES}
        crop_file = _resolve_config_path(args.crop)   # the file load_crop_config read
    except (FileNotFoundError, ValueError) as exc:
        print(f"!! could not load the crop config: {exc}")
        return 1
    crop_sha = sha256_file(crop_file)
    split_sha = sha256_file(split_path)

    # --limit truncates the list *before* the mesh lookup, so a subject this run
    # was never going to reach is not reported as missing.
    selected = wanted_ids[: args.limit] if args.limit > 0 else wanted_ids
    try:
        available = set(list_subjects(root))
    except FileNotFoundError as exc:
        print(f"!! {exc}")
        return 1
    missing = [sid for sid in selected if sid not in available]
    subjects = [sid for sid in selected if sid in available]

    print(f"root: {root}")
    print(f"subject list: {split_path}  ({len(wanted_ids)} IDs, sha256 {split_sha[:16]}...)")
    print(f"crop config: {crop_file}  (sha256 {crop_sha[:16]}...)")
    print(f"  {describe_crop_config(Path(args.crop))}")
    print(f"out: {out_dir}   n_points {args.n_points}   seed {args.seed}   "
          f"targets {'yes' if args.with_targets else 'no'}   "
          f"overwrite {'yes' if args.overwrite else 'no'}   workers {args.workers}")
    print(f"subjects to cache: {len(subjects)}"
          + (f"  (--limit {args.limit})" if args.limit > 0 else ""))
    if missing:
        print(f"  !! {len(missing)} listed subject(s) have no mesh under {root}: "
              f"{missing[:10]}")
    if not subjects and not missing:
        print("!! no subjects found")
        return 1

    summary = RunSummary()
    # A listed subject with no mesh is a failure, not a footnote: a wrong --root
    # or a stale list must not produce a half-empty cache that looks complete.
    for sid in missing:
        for side in SIDES:
            summary.add(EarResult(sid, side, STATUS_FAILED, error=f"no mesh under {root}"))

    jobs = [
        Job(
            subject_id=sid, root=root, out_dir=out_dir, configs=configs,
            n_points=args.n_points, seed=args.seed, with_targets=args.with_targets,
            overwrite=args.overwrite, crop_config_sha256=crop_sha,
            split_file=Path(args.subject_list).as_posix(), split_sha256=split_sha,
        )
        for sid in subjects
    ]
    if jobs:
        out_dir.mkdir(parents=True, exist_ok=True)
        _run(jobs, args.workers, summary)
        # Only after the run: a sweep before it could delete the in-flight temp
        # file of another preprocess run sharing the same --out.
        stray = sweep_temp_files(out_dir)
        if stray:
            print(f"  removed {len(stray)} temp file(s) left by interrupted writes")

    print_summary(summary, out_dir, args.with_targets)
    return 0 if summary.ok else 1


if __name__ == "__main__":
    sys.exit(main())
