#!/usr/bin/env python3
"""Aggregate independently audited pairs by conjunction, without averaging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def aggregate_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine independent pair decisions by conjunction, never by averaging."""

    require(bool(pairs), "no pair reports")
    labels = [str(pair["label"]) for pair in pairs]
    require(len(labels) == len(set(labels)), "duplicate pair labels")
    for pair in pairs:
        for tier in ("formal", "scientific_equivalence"):
            decision = pair["tiers"][tier]["decision"]
            require(
                decision in {"accept", "reject"},
                f"invalid {tier} decision for {pair['label']}: {decision}",
            )
    formal_accept_count = sum(pair["tiers"]["formal"]["decision"] == "accept" for pair in pairs)
    science_accept_count = sum(pair["tiers"]["scientific_equivalence"]["decision"] == "accept" for pair in pairs)
    return {
        "rule": "all-pair conjunction; no averaging",
        "pair_count": len(pairs),
        "formal_accept_pair_count": formal_accept_count,
        "formal_decision": ("accept" if formal_accept_count == len(pairs) else "reject"),
        "scientific_equivalence_accept_pair_count": science_accept_count,
        "scientific_equivalence_decision": ("accept" if science_accept_count == len(pairs) else "reject"),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Default-512 versus threshold-128 multi-pair validation",
        "",
        f"Formal aggregate: **{report['aggregate']['formal_decision'].upper()}**. Scientific-equivalence aggregate: **{report['aggregate']['scientific_equivalence_decision'].upper()}**.",
        "",
        "Both aggregate decisions are all-pair conjunctions. No averaging can hide a failed pair, and the scientific-equivalence tier cannot rewrite a formal outcome.",
        "",
        "| Pair | Formal | Science | Assign mismatches | Poses exact | No collapse | Min FSC-AUC | Max relL2 | HBM ratio | Sparse-wall ratio |",
        "| --- | --- | --- | ---: | :---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in report["pairs"]:
        summary = pair["summary"]
        performance = pair["performance"]
        tiers = pair["tiers"]
        lines.append(
            f"| {pair['label']} | {tiers['formal']['decision'].upper()} | "
            f"{tiers['scientific_equivalence']['decision'].upper()} | "
            f"{summary['assignment_mismatch_count']} | {summary['poses_exact']} | "
            f"{summary['no_class_collapse']} | "
            f"{summary['min_final_map_signed_non_dc_fsc_auc']:.12f} | "
            f"{summary['max_final_map_relative_l2']:.9g} | "
            f"{performance['sampled_hbm_ratio']:.6f} | "
            f"{performance['sparse_group_wall_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Seed 42001 remains governed formally by its earlier prospective contract; its scientific label is retrospective. The additional pair labels are governed prospectively by the attempt-2 contract.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--pair-report", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    contract_hash = sha256(args.contract)
    require(
        contract_hash == args.expected_contract_sha256,
        "contract hash differs from --expected-contract-sha256",
    )
    pairs: list[dict[str, Any]] = []
    labels: set[str] = set()
    pair_hashes: dict[str, str] = {}
    for path in args.pair_report:
        require(path.is_file(), f"missing pair report {path}")
        pair = read_json(path)
        require(pair["contract_sha256"] == contract_hash, f"contract mismatch in {path}")
        label = str(pair["label"])
        require(label not in labels, f"duplicate pair label {label}")
        labels.add(label)
        pair_hashes[str(path.resolve())] = sha256(path)
        pairs.append(pair)
    pairs.sort(key=lambda row: row["label"])
    aggregate = aggregate_pairs(pairs)
    report = {
        "schema": "recovar.em.kclass_default_threshold_pair_aggregate.v1",
        "contract": str(args.contract.resolve()),
        "contract_sha256": contract_hash,
        "formal_thresholds": read_json(args.contract)["formal_thresholds"],
        "scientific_equivalence_thresholds": read_json(args.contract)["scientific_equivalence_thresholds"],
        "pair_report_sha256": pair_hashes,
        "pairs": pairs,
        "aggregate": aggregate,
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(markdown(report))
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    raise SystemExit(
        0
        if aggregate["formal_decision"] == "accept" and aggregate["scientific_equivalence_decision"] == "accept"
        else 3
    )


if __name__ == "__main__":
    main()
