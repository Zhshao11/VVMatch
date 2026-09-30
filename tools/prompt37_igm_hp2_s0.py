#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


SALT = "P37-20260926-IGM-M1-HP2-EVAL|"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    parent_hashes = json.loads((args.parent / "MANIFEST_SHA256.json").read_text())
    for name, digest in parent_hashes.items():
        if file_sha(args.parent / name) != digest:
            raise RuntimeError(f"parent S0 changed: {name}")
    if file_sha(args.parent / "MANIFEST_SHA256.json") != "9718b1a04da78d061ecee010477f4f16e6d7a3c570caa0d7b61075e745149952":
        raise RuntimeError("parent S0 bundle is not the frozen HP1 bundle")

    old_sequences = read_csv(args.parent / "sequence_roles.csv")
    old_files = read_csv(args.parent / "file_manifest.csv")
    old_native = read_csv(args.parent / "native_pairs.csv")
    old_role = {row["sequence"]: row["role"] for row in old_sequences}
    moved_to_eval = set()
    for prefix in ("i_", "v_"):
        candidates = [row["sequence"] for row in old_sequences if row["role"] == "train" and row["sequence"].startswith(prefix)]
        moved_to_eval.update(sorted(candidates, key=lambda value: sha(SALT + value))[:9])
    new_role = {}
    for sequence, role in old_role.items():
        if role == "development":
            new_role[sequence] = "development"
        elif sequence in moved_to_eval:
            new_role[sequence] = "evaluation"
        else:
            new_role[sequence] = "train"

    sequence_rows = []
    for row in sorted(old_sequences, key=lambda value: value["sequence"]):
        sequence_rows.append({"sequence": row["sequence"], "subset": row["subset"], "role": new_role[row["sequence"]], "split_sha256": sha(SALT + row["sequence"]), "parent_role": row["role"]})
    file_rows = [{**row, "role": new_role[row["sequence"]], "parent_role": row["role"]} for row in old_files]
    native_rows = [{**row, "role": new_role[row["sequence"]], "parent_role": row["role"]} for row in old_native]
    directed = []
    for row in native_rows:
        sequence = row["sequence"]; role = row["role"]
        source_name = Path(row["source_image"]).stem
        target_name = Path(row["target_image"]).stem
        directions = [(source_name, target_name, "forward", False)]
        if role == "train":
            directions.append((target_name, source_name, "reverse", True))
        for source, target, direction, invert in directions:
            directed.append({"sample_id": f"{sequence}:{source}->{target}", "native_pair_id": row["pair_id"], "sequence": sequence, "subset": row["subset"], "role": role, "direction": direction, "source_image": f"{sequence}/{source}.ppm", "target_image": f"{sequence}/{target}.ppm", "homography": row["homography"], "invert_homography": invert})

    roles = ("train", "development", "evaluation")
    role_sequences = {role: {row["sequence"] for row in sequence_rows if row["role"] == role} for role in roles}
    role_files = {role: {row["relative_path"] for row in file_rows if row["role"] == role} for role in roles}
    counts = Counter((row["role"], row["subset"]) for row in sequence_rows)
    checks = {
        "sequence_counts_72_18_18": [len(role_sequences[role]) for role in roles] == [72, 18, 18],
        "strata_counts": counts == {("train", "illumination"): 34, ("train", "viewpoint"): 38, ("development", "illumination"): 9, ("development", "viewpoint"): 9, ("evaluation", "illumination"): 9, ("evaluation", "viewpoint"): 9},
        "old_development_preserved": role_sequences["development"] == {row["sequence"] for row in old_sequences if row["role"] == "development"},
        "evaluation_from_old_train_only": role_sequences["evaluation"] == moved_to_eval and all(old_role[sequence] == "train" for sequence in role_sequences["evaluation"]),
        "old_evaluation_moved_to_train": all(new_role[row["sequence"]] == "train" for row in old_sequences if row["role"] == "evaluation"),
        "sequence_intersections_empty": not (role_sequences["train"] & role_sequences["development"] or role_sequences["train"] & role_sequences["evaluation"] or role_sequences["development"] & role_sequences["evaluation"]),
        "file_intersections_empty": not (role_files["train"] & role_files["development"] or role_files["train"] & role_files["evaluation"] or role_files["development"] & role_files["evaluation"]),
        "native_pair_counts": Counter(row["role"] for row in native_rows) == {"train": 360, "development": 90, "evaluation": 90},
        "directed_pair_counts": Counter(row["role"] for row in directed) == {"train": 720, "development": 90, "evaluation": 90},
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    write_csv(args.output / "sequence_roles.csv", sequence_rows)
    write_csv(args.output / "file_manifest.csv", file_rows)
    write_csv(args.output / "native_pairs.csv", native_rows)
    write_csv(args.output / "directed_role_pairs.csv", directed)
    audit = {"experiment": "P37-20260926-IGM-M1-HP2", "status": "S0_PASSED_EVALUATION_UNREAD", "salt": SALT, "parent_bundle_sha256": file_sha(args.parent / "MANIFEST_SHA256.json"), "moved_to_evaluation": sorted(moved_to_eval), "checks": checks}
    (args.output / "S0_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    hashes = {path.name: file_sha(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "MANIFEST_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
