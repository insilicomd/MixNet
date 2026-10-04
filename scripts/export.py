"""Create a byte-reproducible source ZIP from FILES.txt, never a recursive tree export."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path, PurePosixPath
import zipfile


def export(root: Path, output: Path) -> Path:
    names = (root / "FILES.txt").read_text(encoding="utf-8").splitlines()
    if not names or names != sorted(set(names)) or "FILES.txt" not in names:
        raise ValueError("FILES.txt must be a sorted unique explicit file allowlist including itself")
    payloads: dict[str, bytes] = {}
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError(f"unsafe inventory path: {name}")
        source = root / name
        if any((root / Path(*path.parts[:i])).is_symlink() for i in range(1, len(path.parts) + 1)):
            raise ValueError(f"symlinks are forbidden: {name}")
        if not source.is_file():
            raise ValueError(f"missing inventory file: {name}")
        payloads[name] = source.read_bytes()
    checksums = "".join(f"{sha256(payload).hexdigest()}  {name}\n"
                        for name, payload in payloads.items()).encode()
    if "SHA256SUMS" in payloads:
        raise ValueError("SHA256SUMS is generated, not a source inventory member")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "mixture-lco-0.1.0-source.zip"
    # Stored ZIP entries have fixed metadata, eliminating compressor/platform differences.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as stream:
        for name, payload in sorted({**payloads, "SHA256SUMS": checksums}.items()):
            entry = zipfile.ZipInfo(f"mixture-lco-0.1.0/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            stream.writestr(entry, payload)
    (output / "SHA256SUMS").write_bytes(checksums)
    (output / "ARCHIVE_SHA256SUMS").write_text(
        f"{sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n", encoding="utf-8")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(Path(__file__).resolve().parents[1], args.output))


if __name__ == "__main__":
    main()
