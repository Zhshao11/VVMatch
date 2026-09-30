#!/usr/bin/env python3
"""Freeze the HPatches role split for P37-20260921-IGM-M0-HP1.

This stage reads only the dataset archive and the E-051 protocol constants. It
does not load model outputs, evaluation tables, or historical metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


EXPERIMENT_ID = "P37-20260921-IGM-M0-HP1"
SPLIT_SALT = "P37-20260921-IGM-HP1|"
IGNORED = {
    "i_contruction",
    "i_crownnight",
    "i_dc",
    "i_pencils",
    "i_whitebuilding",
    "v_artisans",
    "v_astronautis",
    "v_talent",
}
ROLE_COUNTS = {"i": (34, 9, 9), "v": (38, 9, 9)}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_members(archive: zipfile.ZipFile) -> tuple[str, dict[str, zipfile.ZipInfo]]:
    files = {item.filename: item for item in archive.infolist() if not item.is_dir()}
    roots = {name.split("/", 1)[0] for name in files}
    if len(roots) != 1:
        raise RuntimeError(f"expected one archive root, got {sorted(roots)}")
    return next(iter(roots)), files


def split_sequences(sequences: list[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for prefix in ("i", "v"):
        stratum = sorted(
            (seq for seq in sequences if seq.startswith(prefix + "_")),
            key=lambda seq: stable_hash(SPLIT_SALT + seq),
        )
        train_n, dev_n, eval_n = ROLE_COUNTS[prefix]
        if len(stratum) != train_n + dev_n + eval_n:
            raise RuntimeError(f"unexpected {prefix} stratum count: {len(stratum)}")
        for seq in stratum[:train_n]:
            roles[seq] = "train"
        for seq in stratum[train_n : train_n + dev_n]:
            roles[seq] = "development"
        for seq in stratum[train_n + dev_n :]:
            roles[seq] = "evaluation"
    return roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"artifact root is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as archive:
        root, members = archive_members(archive)
        all_sequences = sorted(
            {
                name.split("/")[1]
                for name in members
                if len(name.split("/")) >= 3
                and name.split("/")[1].startswith(("i_", "v_"))
            }
        )
        sequences = [seq for seq in all_sequences if seq not in IGNORED]
        roles = split_sequences(sequences)

        sequence_rows: list[dict[str, object]] = []
        file_rows: list[dict[str, object]] = []
        native_rows: list[dict[str, object]] = []
        directed_rows: list[dict[str, object]] = []
        for seq in sequences:
            role = roles[seq]
            subset = "illumination" if seq.startswith("i_") else "viewpoint"
            sequence_rows.append(
                {
                    "sequence": seq,
                    "subset": subset,
                    "role": role,
                    "split_sha256": stable_hash(SPLIT_SALT + seq),
                }
            )
            required = [f"{idx}.ppm" for idx in range(1, 7)] + [
                f"H_1_{idx}" for idx in range(2, 7)
            ]
            for basename in required:
                archive_name = f"{root}/{seq}/{basename}"
                if archive_name not in members:
                    raise FileNotFoundError(archive_name)
                payload = archive.read(archive_name)
                file_rows.append(
                    {
                        "sequence": seq,
                        "role": role,
                        "relative_path": f"{seq}/{basename}",
                        "size_bytes": len(payload),
                        "sha256": sha256_bytes(payload),
                    }
                )
            for target in range(2, 7):
                pair_id = f"{seq}:1->{target}"
                native_rows.append(
                    {
                        "pair_id": pair_id,
                        "sequence": seq,
                        "subset": subset,
                        "role": role,
                        "source_image": f"{seq}/1.ppm",
                        "target_image": f"{seq}/{target}.ppm",
                        "homography": f"{seq}/H_1_{target}",
                    }
                )
                directions = ((1, target, "forward"),)
                if role == "train":
                    directions += ((target, 1, "reverse"),)
                for source, destination, direction in directions:
                    directed_rows.append(
                        {
                            "sample_id": f"{seq}:{source}->{destination}",
                            "native_pair_id": pair_id,
                            "sequence": seq,
                            "subset": subset,
                            "role": role,
                            "direction": direction,
                            "source_image": f"{seq}/{source}.ppm",
                            "target_image": f"{seq}/{destination}.ppm",
                            "homography": f"{seq}/H_1_{target}",
                            "invert_homography": direction == "reverse",
                        }
                    )

    role_sets = {
        role: {row["sequence"] for row in sequence_rows if row["role"] == role}
        for role in ("train", "development", "evaluation")
    }
    intersections = {
        "train_development": sorted(role_sets["train"] & role_sets["development"]),
        "train_evaluation": sorted(role_sets["train"] & role_sets["evaluation"]),
        "development_evaluation": sorted(
            role_sets["development"] & role_sets["evaluation"]
        ),
    }
    file_role_sets = {
        role: {row["relative_path"] for row in file_rows if row["role"] == role}
        for role in role_sets
    }
    file_intersections = {
        "train_development": sorted(file_role_sets["train"] & file_role_sets["development"]),
        "train_evaluation": sorted(file_role_sets["train"] & file_role_sets["evaluation"]),
        "development_evaluation": sorted(
            file_role_sets["development"] & file_role_sets["evaluation"]
        ),
    }
    role_counts = Counter(row["role"] for row in sequence_rows)
    subset_counts = Counter((row["role"], row["subset"]) for row in sequence_rows)
    native_counts = Counter(row["role"] for row in native_rows)
    directed_counts = Counter(row["role"] for row in directed_rows)
    checks = {
        "paper_sequence_count_108": len(sequences) == 108,
        "paper_strata_52_56": sum(s.startswith("i_") for s in sequences) == 52
        and sum(s.startswith("v_") for s in sequences) == 56,
        "role_counts_72_18_18": role_counts
        == {"train": 72, "development": 18, "evaluation": 18},
        "stratified_counts": dict(subset_counts)
        == {
            ("train", "illumination"): 34,
            ("train", "viewpoint"): 38,
            ("development", "illumination"): 9,
            ("development", "viewpoint"): 9,
            ("evaluation", "illumination"): 9,
            ("evaluation", "viewpoint"): 9,
        },
        "native_pair_counts_360_90_90": native_counts
        == {"train": 360, "development": 90, "evaluation": 90},
        "directed_train_count_720": directed_counts["train"] == 720,
        "sequence_intersections_empty": not any(intersections.values()),
        "file_intersections_empty": not any(file_intersections.values()),
        "required_file_count_1188": len(file_rows) == 108 * 11,
    }
    if not all(checks.values()):
        raise RuntimeError(f"S0 gate failed: {checks}")

    write_csv(args.output / "sequence_roles.csv", sequence_rows)
    write_csv(args.output / "file_manifest.csv", file_rows)
    write_csv(args.output / "native_pairs.csv", native_rows)
    write_csv(args.output / "directed_role_pairs.csv", directed_rows)
    audit = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "status": "S0_PASSED",
        "source": {
            "zip": str(args.zip.resolve()),
            "zip_size_bytes": args.zip.stat().st_size,
            "zip_sha256": sha256_file(args.zip),
            "archive_root": root,
            "allowlist_source": "E-051 LoFTR/DKM ignore-list constant; no metric payload read",
            "ignored_sequences": sorted(IGNORED),
        },
        "split": {
            "salt": SPLIT_SALT,
            "rule": "sort each i_/v_ stratum by SHA256(salt + sequence)",
            "role_counts": dict(role_counts),
            "subset_counts": {f"{role}:{subset}": count for (role, subset), count in subset_counts.items()},
            "native_pair_counts": dict(native_counts),
            "directed_role_pair_counts": dict(directed_counts),
        },
        "intersections": intersections,
        "file_intersections": file_intersections,
        "checks": checks,
    }
    (args.output / "S0_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(args.output.iterdir())
        if path.is_file()
    }
    (args.output / "MANIFEST_SHA256.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
