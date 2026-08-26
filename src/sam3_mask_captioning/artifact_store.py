from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

from .io_utils import sha256_file


def _safe_member(member: tarfile.TarInfo, destination: Path) -> None:
    resolved = (destination / member.name).resolve()
    try:
        resolved.relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe archive member: {member.name}") from exc
    if member.issym() or member.islnk():
        raise ValueError(f"Links are not allowed in artifact archives: {member.name}")


def hydrate_archive(archive: str | Path, destination: str | Path) -> list[Path]:
    archive = Path(archive)
    destination = Path(destination)
    if not archive.exists():
        raise FileNotFoundError(archive)
    extracted: list[Path] = []
    with tarfile.open(archive, "r") as handle:
        for member in handle.getmembers():
            _safe_member(member, destination)
            handle.extract(member, destination)
            if member.isfile():
                extracted.append(destination / member.name)
    return extracted


def hydrate_archive_members(
    archive: str | Path,
    destination: str | Path,
    member_names: Iterable[str | Path],
) -> list[Path]:
    """Extract only missing requested files from an artifact archive.

    Caption-only iterations often select one image from a 100-image campaign
    unit. Expanding the whole source/SAM3 archive in that case multiplies
    shared-filesystem reads and small-file creation for no inference benefit.
    This helper validates the complete request before writing anything, reads
    requested members in physical tar order, preserves files already present,
    and returns only paths created by this call so cleanup cannot remove
    pre-existing artifacts.
    """
    archive = Path(archive)
    destination = Path(destination)
    if not archive.exists():
        raise FileNotFoundError(archive)

    requested = {
        Path(value).as_posix()
        for value in member_names
        if str(value).strip()
    }
    if not requested:
        return []
    if any(
        not name or Path(name).is_absolute() or ".." in Path(name).parts
        for name in requested
    ):
        raise ValueError("Artifact member names must be safe nonempty relative paths")

    extracted: list[Path] = []
    with tarfile.open(archive, "r") as handle:
        members_by_name = {member.name: member for member in handle.getmembers()}
        missing = sorted(requested - members_by_name.keys())
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} requested member(s) in {archive}: {missing[:3]}"
            )
        selected = [members_by_name[name] for name in requested]
        for member in selected:
            _safe_member(member, destination)
            if not member.isfile():
                raise ValueError(
                    f"Requested artifact member is not a regular file: {member.name}"
                )
        for member in sorted(selected, key=lambda value: value.offset):
            target = destination / member.name
            if target.is_symlink():
                raise ValueError(f"Refusing existing symlink artifact target: {target}")
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"Artifact target is not a file: {target}")
                continue
            handle.extract(member, destination)
            extracted.append(target)
    return extracted


def pack_artifacts(
    root: str | Path,
    relative_paths: Iterable[str | Path],
    archive: str | Path,
) -> dict[str, object]:
    root = Path(root).resolve()
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{archive.name}.", suffix=".tmp", dir=archive.parent
    )
    os.close(fd)
    members: list[str] = []
    try:
        with tarfile.open(temporary, "w") as handle:
            for value in relative_paths:
                path = (root / value).resolve()
                try:
                    relative = path.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"Artifact is outside root: {path}") from exc
                if not path.exists():
                    continue
                if path.is_dir():
                    files = sorted(item for item in path.rglob("*") if item.is_file())
                else:
                    files = [path]
                for item in files:
                    name = item.relative_to(root).as_posix()
                    handle.add(item, arcname=name, recursive=False)
                    members.append(name)
        os.replace(temporary, archive)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {
        "archive_path": str(archive),
        "sha256": sha256_file(archive),
        "member_count": len(members),
        "members": members,
    }


def remove_hydrated_files(root: str | Path, relative_paths: Iterable[str | Path]) -> None:
    """Remove only known hydrated files/directories inside one campaign unit.

    A campaign worker may expose a node-local hydrated directory through a
    top-level symlink in the unit. Unlink that managed entry without following
    it; ordinary files and directories retain the stricter resolved-path check.
    """
    root = Path(root).resolve()
    for value in relative_paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Cleanup target must be relative to unit root: {value}")
        lexical_target = root / relative
        if lexical_target.is_symlink():
            lexical_target.unlink()
            continue
        target = lexical_target.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Cleanup target is outside unit root: {target}") from exc
        if not target.exists():
            continue
        if target.is_file():
            target.unlink()
            continue
        for item in sorted(target.rglob("*"), reverse=True):
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        target.rmdir()


def read_tar_member(archive: str | Path, member_name: str) -> bytes:
    with tarfile.open(archive, "r") as handle:
        member = handle.getmember(member_name)
        extracted = handle.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Missing tar member {member_name} in {archive}")
        return extracted.read()
