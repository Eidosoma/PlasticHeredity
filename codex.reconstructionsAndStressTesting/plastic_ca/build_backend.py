"""Minimal no-network build backend for this pure-Python project."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import pathlib
import zipfile


def _metadata() -> bytes:
    return (
        "Metadata-Version: 2.1\n"
        "Name: plastic-ca-cleanroom\n"
        "Version: 0.1.0\n"
        "Requires-Python: >=3.11\n"
        "Requires-Dist: numpy==2.5.2\n\n"
    ).encode()


def build_wheel(wheel_directory: str, config_settings=None, metadata_directory=None) -> str:
    root = pathlib.Path(__file__).resolve().parent.parent
    name = "plastic_ca_cleanroom-0.1.0-py3-none-any.whl"
    target = pathlib.Path(wheel_directory) / name
    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        package_root = root / "plastic_ca"
        for path in sorted(
            candidate
            for candidate in package_root.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".json"}
        ):
            arc = f"plastic_ca/{path.relative_to(package_root).as_posix()}"
            data = path.read_bytes()
            archive.writestr(arc, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((arc, f"sha256={digest}", str(len(data))))
        dist = "plastic_ca_cleanroom-0.1.0.dist-info"
        files = {
            f"{dist}/METADATA": _metadata(),
            f"{dist}/WHEEL": b"Wheel-Version: 1.0\nGenerator: cleanroom\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            f"{dist}/entry_points.txt": b"[console_scripts]\nplastic-ca = plastic_ca.cli:main\n",
        }
        for arc, data in files.items():
            archive.writestr(arc, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((arc, f"sha256={digest}", str(len(data))))
        record_path = f"{dist}/RECORD"
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerows(records + [(record_path, "", "")])
        archive.writestr(record_path, buf.getvalue().encode())
    return name


def prepare_metadata_for_build_wheel(metadata_directory: str, config_settings=None) -> str:
    dist = "plastic_ca_cleanroom-0.1.0.dist-info"
    path = pathlib.Path(metadata_directory) / dist
    path.mkdir(parents=True, exist_ok=True)
    (path / "METADATA").write_bytes(_metadata())
    return dist


def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []
