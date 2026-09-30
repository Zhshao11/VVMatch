#!/usr/bin/env python3
"""Launch reviewed M0R1 GPU stages only from the approved runtime closure."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


APPROVED_RUNTIME_MANIFEST_SHA256 = "789564cc96520477108e42dc451571fb0d814d7ea09e964cb435ef43de951d4b"
EXPECTED_RUNNER = "prompt37_igm_m0r1_hp1_runner.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-source-hashes", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    digest = sha256_file(args.approved_source_hashes)
    if digest != APPROVED_RUNTIME_MANIFEST_SHA256:
        raise RuntimeError("runtime source manifest is not the reviewer-approved closure")
    expected_runner_path = (Path(__file__).resolve().parent / EXPECTED_RUNNER).resolve()
    if args.runner.resolve() != expected_runner_path or not args.runner.is_file() or not args.python.is_file():
        raise RuntimeError("launch guard runner or Python identity mismatch")
    records = {}
    for line in args.approved_source_hashes.read_text(encoding="utf-8").splitlines():
        file_digest, relative = line.split(maxsplit=1)
        records[relative] = file_digest
    runner_role = f"tools/{EXPECTED_RUNNER}"
    if records.get(runner_role) != sha256_file(args.runner):
        raise RuntimeError("launch runner bytes differ from the approved runtime manifest")
    runner_args = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    if "--approved-source-hashes" not in runner_args:
        raise RuntimeError("runner must receive the same approved source manifest")
    supplied = Path(runner_args[runner_args.index("--approved-source-hashes") + 1]).resolve()
    if supplied != args.approved_source_hashes.resolve():
        raise RuntimeError("runner approved-source path differs from launch guard")
    environment = os.environ.copy()
    environment["P37_M0R1_SOURCE_ANCHOR_VERIFIED"] = digest
    environment["P37_M0R1_GUARD_PID"] = str(os.getpid())
    completed = subprocess.run(
        [str(args.python.resolve()), str(args.runner.resolve()), *runner_args],
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
