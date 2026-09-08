"""Batch preprocessing: raw Huawei PLYs -> cached canonical ears (Role A).

PLACEHOLDER. Not implemented yet.

Loaders in ``src/data.py`` are done. Blocked on the frozen crop bounds:
run ``scripts/crop_stats.py`` on the training annotations to produce
``configs/crop.yaml`` (then ``load_crop_config`` in ``src/geometry.py`` reads it).

Planned behaviour once unblocked::

    for subject in list_subjects(DATA_ROOT):
        raw = load_mesh(mesh_path(subject))
        for side in ("left", "right"):
            ear = canonicalize_ear(raw, side, load_crop_config(side))
            save <cache>/<subject>_<side>.npz   # points, normals, transform, qa
    write a QA summary over all subjects (vertex counts, bboxes, suspicious flags)

The cache is NDA-protected and stays out of Git (see ``.gitignore``).
"""

from __future__ import annotations

import sys

TODO = (
    "TODO: scripts/preprocess.py is a placeholder.\n"
    "Blocked on:\n"
    "  1. configs/crop.yaml - run scripts/crop_stats.py on the training annotations\n"
)


def main() -> int:
    sys.stderr.write(TODO)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
