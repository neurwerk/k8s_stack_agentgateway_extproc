"""Verify that a release tag exactly matches the Python project version."""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from re import Pattern, compile

PROJECT_FILE = Path(__file__).resolve().parents[1] / "pyproject.toml"
LOCK_FILE = PROJECT_FILE.with_name("uv.lock")
TAG_PATTERN: Pattern[str] = compile(r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\Z")


class ReleaseVerificationError(ValueError):
    """Indicate invalid project version metadata or a mismatched release tag."""

    @classmethod
    def missing_project_table(cls) -> ReleaseVerificationError:
        """Create an error for missing project metadata."""
        return cls("pyproject.toml has no [project] table")

    @classmethod
    def missing_project_version(cls) -> ReleaseVerificationError:
        """Create an error for a missing static version."""
        return cls("pyproject.toml has no static project.version")

    @classmethod
    def missing_project_name(cls) -> ReleaseVerificationError:
        """Create an error for a missing project name."""
        return cls("pyproject.toml has no project.name")

    @classmethod
    def invalid_tag(cls, tag: str) -> ReleaseVerificationError:
        """Create an error for a non-release tag."""
        return cls(f"release tag {tag!r} must match vX.Y.Z")

    @classmethod
    def missing_lock_version(cls, project_name: str) -> ReleaseVerificationError:
        """Create an error for a missing editable root package in uv.lock."""
        return cls(f"uv.lock has no unique editable root package for {project_name!r}")

    @classmethod
    def lock_mismatch(cls, project_version: str, lock_version: str) -> ReleaseVerificationError:
        """Create an error for inconsistent version sources."""
        return cls(
            "pyproject.toml version "
            f"{project_version!r} does not match uv.lock version {lock_version!r}"
        )

    @classmethod
    def tag_mismatch(cls, tag: str, expected: str) -> ReleaseVerificationError:
        """Create an error for a release tag mismatch."""
        return cls(f"release tag {tag!r} does not match project version {expected!r}")


def load_project_version(project_file: Path) -> str:
    """Read the static PEP 621 project version."""
    return load_project_metadata(project_file)[1]


def load_project_metadata(project_file: Path) -> tuple[str, str]:
    """Read the static PEP 621 project name and version."""
    with project_file.open("rb") as file:
        document = tomllib.load(file)
    project = document.get("project")
    if not isinstance(project, dict):
        raise ReleaseVerificationError.missing_project_table()
    name = project.get("name")
    if not isinstance(name, str) or not name:
        raise ReleaseVerificationError.missing_project_name()
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ReleaseVerificationError.missing_project_version()
    return name, version


def load_lock_version(lock_file: Path, project_name: str) -> str:
    """Read the version of the editable root package from uv.lock."""
    with lock_file.open("rb") as file:
        document = tomllib.load(file)
    packages = document.get("package")
    if not isinstance(packages, list):
        raise ReleaseVerificationError.missing_lock_version(project_name)
    versions = []
    for package in packages:
        if not isinstance(package, dict) or package.get("name") != project_name:
            continue
        source = package.get("source")
        if isinstance(source, dict) and source.get("editable") == ".":
            versions.append(package.get("version"))
    if len(versions) != 1 or not isinstance(versions[0], str) or not versions[0]:
        raise ReleaseVerificationError.missing_lock_version(project_name)
    return versions[0]


def verify_release_tag(
    tag: str, project_file: Path = PROJECT_FILE, lock_file: Path = LOCK_FILE
) -> str:
    """Return the expected tag or reject a mismatch."""
    if TAG_PATTERN.fullmatch(tag) is None:
        raise ReleaseVerificationError.invalid_tag(tag)
    project_name, project_version = load_project_metadata(project_file)
    lock_version = load_lock_version(lock_file, project_name)
    if project_version != lock_version:
        raise ReleaseVerificationError.lock_mismatch(project_version, lock_version)
    expected = f"v{project_version}"
    if tag != expected:
        raise ReleaseVerificationError.tag_mismatch(tag, expected)
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    """Run the release-tag check from the command line."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        sys.stderr.write("usage: verify_release_tag.py <tag>\n")
        return 2
    try:
        verified = verify_release_tag(arguments[0])
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        sys.stderr.write(f"release version check failed: {exc}\n")
        return 1
    sys.stdout.write(f"release tag matches project version: {verified}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
