"""Rebuild the artifact manifest: one SHA-256 per file, with the run matrix.

The manifest exists so that a reader can verify the committed evidence without
re-running anything, and so that each campaign's *matrix* (what was varied) is
recorded next to its *checksums* (what was produced). Campaign notes live in
``manifest-notes.json`` because a note is authored, not derived; everything else
is scanned from disk.

Usage
-----
    python scripts/probes/asym_apc_native/build_artifacts_manifest.py \
      --artifacts docs/experiments/asym-apc-independent-validation-artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "kvopt.asym_apc.independent_validation_manifest.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build(root: Path) -> dict[str, object]:
    notes_path = root / "manifest-notes.json"
    notes: dict[str, str] = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    campaigns = []
    for campaign_dir in sorted((root / "campaigns").iterdir()):
        manifest_path = campaign_dir / "campaign-manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matrix = dict(manifest["matrix"])
        files = [
            _entry(root, path)
            for path in sorted(campaign_dir.rglob("*"))
            if path.is_file()
        ]
        campaigns.append(
            {
                "name": campaign_dir.name,
                "note": notes.get(campaign_dir.name, str(manifest.get("label", ""))),
                "cells": len(manifest.get("cells", [])),
                "matrix": matrix,
                "calibration": manifest["calibration"],
                "files": files,
            }
        )
    calibration = [
        _entry(root, path)
        for path in sorted((root / "calibration").rglob("*"))
        if path.is_file()
    ]
    analysis = [
        _entry(root, path)
        for path in sorted((root / "analysis").rglob("*"))
        if path.is_file()
    ]
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_host": "sandbox CPU device (no CUDA device present)",
        "campaigns": campaigns,
        "calibration_files": calibration,
        "analysis_files": analysis,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--check", action="store_true",
                        help="verify the existing manifest instead of rewriting it")
    args = parser.parse_args(argv)
    root = Path(args.artifacts).resolve()
    document = build(root)
    if args.check:
        existing = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        old = {e["path"]: e["sha256"] for group in
               (existing["campaigns"],) for campaign in group for e in campaign["files"]}
        for group in ("calibration_files", "analysis_files"):
            old.update({e["path"]: e["sha256"] for e in existing[group]})
        new = {e["path"]: e["sha256"] for campaign in document["campaigns"]
               for e in campaign["files"]}
        for group in ("calibration_files", "analysis_files"):
            new.update({e["path"]: e["sha256"] for e in document[group]})
        missing = sorted(set(old) - set(new))
        changed = sorted(p for p in set(old) & set(new) if old[p] != new[p])
        print(f"files on disk: {len(new)}, in manifest: {len(old)}, "
              f"missing: {len(missing)}, changed: {len(changed)}")
        for path in (missing + changed)[:10]:
            print("  ", path)
        return 0 if not missing and not changed else 1
    (root / "MANIFEST.json").write_text(
        json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    total = sum(len(c["files"]) for c in document["campaigns"]) + len(
        document["calibration_files"]
    ) + len(document["analysis_files"])
    print(f"wrote MANIFEST.json: {len(document['campaigns'])} campaigns, {total} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
