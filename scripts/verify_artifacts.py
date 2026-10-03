"""Read-only byte-integrity check of the public saved results and scorer files.

Python standard library only. This does not load models, decode arrays, perform
inference, recompute metrics, or reproduce training. It writes no files.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(root):
    manifest_path = root / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "public-saved-artifacts-v1":
        raise ValueError("Unsupported manifest")
    failures, checked = [], 0
    for record in manifest["files"]:
        try:
            token = PurePosixPath(record["path"])
            if token.is_absolute() or ".." in token.parts or "\\" in record["path"] or ":" in record["path"]:
                raise ValueError("Unsafe relative path")
            path = root.joinpath(*token.parts).resolve()
            path.relative_to(root)
            if not path.is_file():
                raise ValueError("Missing artifact")
            if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError("Size or SHA256 mismatch")
            checked += 1
        except (ValueError, OSError, KeyError) as error:
            failures.append({"path": record.get("path"), "error": str(error)})
    return {"passed": not failures, "files_checked": checked,
            "files_expected": len(manifest["files"]), "manifest_sha256": digest(manifest_path),
            "scope": "Sizes and SHA256 bytes only; not model loading or scientific recomputation",
            "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        report = verify(args.root.resolve())
    except (ValueError, KeyError, OSError) as error:
        report = {"passed": False, "error": str(error)}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
