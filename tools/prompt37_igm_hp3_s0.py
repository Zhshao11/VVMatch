#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


SALT = "P37-20260930-GCRP-PROBE|"
HP2_BUNDLE = "273996d03efd2139020e2e9dd1f052e3c74710259276455408c724c05ec1ca4d"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp2-s0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if file_hash(args.hp2_s0 / "MANIFEST_SHA256.json") != HP2_BUNDLE:
        raise RuntimeError("HP2 frozen bundle changed")
    recorded = json.loads((args.hp2_s0 / "MANIFEST_SHA256.json").read_text())
    for name, digest in recorded.items():
        if file_hash(args.hp2_s0 / name) != digest:
            raise RuntimeError(f"HP2 artifact changed: {name}")
    sequences, directed = read(args.hp2_s0 / "sequence_roles.csv"), read(args.hp2_s0 / "directed_role_pairs.csv")
    train = [row for row in sequences if row["role"] == "train"]
    probe = set()
    for subset in ("illumination", "viewpoint"):
        candidates = [row["sequence"] for row in train if row["subset"] == subset]
        probe.update(sorted(candidates, key=lambda name: text_hash(SALT + name))[:4])
    pretrain_rows = []
    for row in train:
        role = "probe" if row["sequence"] in probe else "fit"
        pretrain_rows.append({"sequence": row["sequence"], "subset": row["subset"], "pretrain_role": role, "probe_rank_sha256": text_hash(SALT + row["sequence"]), "hp3_role": row["role"]})
    pair_rows = []
    role_by_sequence = {row["sequence"]: row["pretrain_role"] for row in pretrain_rows}
    for row in directed:
        if row["role"] == "train":
            pair_rows.append({**row, "pretrain_role": role_by_sequence[row["sequence"]]})
    fit_sequences = {row["sequence"] for row in pretrain_rows if row["pretrain_role"] == "fit"}
    probe_sequences = {row["sequence"] for row in pretrain_rows if row["pretrain_role"] == "probe"}
    dev_sequences = {row["sequence"] for row in sequences if row["role"] == "development"}
    eval_sequences = {row["sequence"] for row in sequences if row["role"] == "evaluation"}
    counts = Counter((row["pretrain_role"], row["subset"]) for row in pretrain_rows)
    checks = {
        "hp2_roles_reused_exactly": len(sequences) == 108 and Counter(row["role"] for row in sequences) == {"train": 72, "development": 18, "evaluation": 18},
        "fit_probe_counts_64_8": len(fit_sequences) == 64 and len(probe_sequences) == 8,
        "probe_strata_4_4": counts[("probe", "illumination")] == 4 and counts[("probe", "viewpoint")] == 4,
        "fit_probe_disjoint": fit_sequences.isdisjoint(probe_sequences),
        "fit_probe_dev_eval_disjoint": (fit_sequences | probe_sequences).isdisjoint(dev_sequences | eval_sequences),
        "fit_probe_pair_partition": Counter(row["pretrain_role"] for row in pair_rows) == {"fit": 640, "probe": 80},
        "development_evaluation_payload_read": False,
        "probe_payload_read": False,
    }
    if not all(value for key, value in checks.items() if not key.endswith("_read")) or checks["development_evaluation_payload_read"] or checks["probe_payload_read"]:
        raise RuntimeError(checks)
    write(args.output / "pretrain_sequence_roles.csv", sorted(pretrain_rows, key=lambda row: row["sequence"]))
    write(args.output / "pretrain_directed_pairs.csv", pair_rows)
    audit = {"experiment": "P37-20260930-IGM-M3-HP3", "status": "S0_PASSED_PROBE_DEV_EVAL_UNREAD", "parent_hp2_bundle_sha256": HP2_BUNDLE, "salt": SALT, "checks": checks, "probe_sequences": sorted(probe_sequences)}
    (args.output / "S0_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    hashes = {path.name: file_hash(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "MANIFEST_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
