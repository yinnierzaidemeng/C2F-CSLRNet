# SPDX-License-Identifier: MIT
"""Check a downloaded public dataset using the Python standard library."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    path = Path(sys.argv[1])
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate archive member")
        for name in names:
            token = PurePosixPath(name)
            require(not token.is_absolute() and ".." not in token.parts and "\\" not in name, "Unsafe member name")
        require(archive.testzip() is None, "ZIP CRC mismatch")
        matches = [name for name in names if name.endswith("/DATA_MANIFEST.json")]
        require(len(matches) == 1, "Dataset manifest missing")
        prefix = matches[0][:-len("DATA_MANIFEST.json")]
        manifest = json.loads(archive.read(matches[0]))
        require(set(names) == {prefix + name for name in manifest["files"]} | {matches[0]}, "Dataset membership mismatch")
        for name, record in manifest["files"].items():
            data = archive.read(prefix + name)
            require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], "Dataset byte mismatch")
        index = json.loads(archive.read(prefix + "PUBLIC_DATA_INDEX.json"))
        for group in index["groups"].values():
            for name in group.values():
                require(prefix + name in names, "Unresolved dataset entry point")
        bindings = json.loads(archive.read(prefix + "SOURCE_BINDINGS.json"))
        for record in bindings["files"]:
            require(manifest["files"][record["path"]]["sha256"] == record["sha256"], "Public-source binding mismatch")
    print(json.dumps({"passed": True, "files": len(manifest["files"]),
                      "version": index["version"], "scope": "Saved bytes and source bindings; no numerical execution"}, indent=2))


if __name__ == "__main__":
    main()
