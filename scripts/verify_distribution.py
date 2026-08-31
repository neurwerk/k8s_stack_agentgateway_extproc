"""Verify license metadata and files in built Python distributions."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LICENSE_EXPRESSION = "MIT AND Apache-2.0"
LICENSE_FILES = (
    Path("LICENSE"),
    Path("LICENSES/Apache-2.0.txt"),
    Path("THIRD_PARTY_NOTICES.md"),
)


class DistributionVerificationError(ValueError):
    """Indicate invalid metadata or license content in a built distribution."""

    @classmethod
    def expected_one(cls, description: str, count: int) -> DistributionVerificationError:
        """Create an error for an unexpected artifact count."""
        return cls(f"expected one {description}, found {count}")

    @classmethod
    def invalid_license_expression(
        cls, source: str, values: list[str]
    ) -> DistributionVerificationError:
        """Create an error for invalid package license metadata."""
        return cls(f"{source} has license expression {values!r}")

    @classmethod
    def different_license_file(
        cls, distribution: str, relative_path: Path
    ) -> DistributionVerificationError:
        """Create an error for changed packaged license content."""
        return cls(f"{distribution} license file differs from source: {relative_path}")

    @classmethod
    def irregular_metadata_file(cls) -> DistributionVerificationError:
        """Create an error for invalid sdist metadata storage."""
        return cls("sdist PKG-INFO is not a regular file")


def _only(paths: Iterable[Path], description: str) -> Path:
    matches = list(paths)
    if len(matches) != 1:
        raise DistributionVerificationError.expected_one(description, len(matches))
    return matches[0]


def _verify_metadata(payload: bytes, source: str) -> None:
    prefix = "License-Expression: "
    values = [
        line.removeprefix(prefix)
        for line in payload.decode().splitlines()
        if line.startswith(prefix)
    ]
    if values != [LICENSE_EXPRESSION]:
        raise DistributionVerificationError.invalid_license_expression(source, values)


def verify_wheel(wheel_path: Path) -> None:
    """Verify wheel metadata and embedded license files."""
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        metadata_name = _only(
            (Path(name) for name in names if name.endswith(".dist-info/METADATA")),
            "wheel METADATA file",
        ).as_posix()
        _verify_metadata(archive.read(metadata_name), metadata_name)
        license_root = metadata_name.removesuffix("METADATA") + "licenses/"
        for relative_path in LICENSE_FILES:
            archive_name = license_root + relative_path.as_posix()
            if archive.read(archive_name) != (PROJECT_ROOT / relative_path).read_bytes():
                raise DistributionVerificationError.different_license_file("wheel", relative_path)


def verify_sdist(sdist_path: Path) -> None:
    """Verify source distribution metadata and embedded license files."""
    with tarfile.open(sdist_path, "r:gz") as archive:
        names = archive.getnames()
        metadata_name = _only(
            (Path(name) for name in names if name.endswith("/PKG-INFO")),
            "sdist PKG-INFO file",
        ).as_posix()
        metadata_file = archive.extractfile(metadata_name)
        if metadata_file is None:
            raise DistributionVerificationError.irregular_metadata_file()
        _verify_metadata(metadata_file.read(), metadata_name)
        for relative_path in LICENSE_FILES:
            suffix = f"/{relative_path.as_posix()}"
            archive_name = _only(
                (Path(name) for name in names if name.endswith(suffix)),
                f"sdist {relative_path} file",
            ).as_posix()
            archived_file = archive.extractfile(archive_name)
            if (
                archived_file is None
                or archived_file.read() != (PROJECT_ROOT / relative_path).read_bytes()
            ):
                raise DistributionVerificationError.different_license_file("sdist", relative_path)


def verify_distribution_directory(directory: Path) -> None:
    """Verify the single wheel and source distribution in a directory."""
    verify_wheel(_only(directory.glob("*.whl"), "wheel"))
    verify_sdist(_only(directory.glob("*.tar.gz"), "source distribution"))


def main(argv: Sequence[str] | None = None) -> int:
    """Run distribution checks from the command line."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        sys.stderr.write("usage: verify_distribution.py <artifact-directory>\n")
        return 2
    try:
        verify_distribution_directory(Path(arguments[0]))
    except (OSError, UnicodeError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        sys.stderr.write(f"distribution verification failed: {exc}\n")
        return 1
    sys.stdout.write("wheel and source distribution license metadata verified\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
