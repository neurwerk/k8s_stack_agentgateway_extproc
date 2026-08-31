from pathlib import Path

import pytest

from scripts.verify_release_tag import PROJECT_FILE, load_project_version, main, verify_release_tag


def _write_release_files(
    path: Path, version: str = "1.2.3", lock_version: str | None = None
) -> tuple[Path, Path]:
    project_file = path / "pyproject.toml"
    project_file.write_text(f'[project]\nname = "example"\nversion = "{version}"\n')
    lock_file = path / "uv.lock"
    lock_file.write_text(
        'version = 1\n\n[[package]]\nname = "example"\n'
        f'version = "{lock_version or version}"\nsource = {{ editable = "." }}\n'
    )
    return project_file, lock_file


def test_release_tag_matches_static_project_version(tmp_path: Path) -> None:
    project_file, lock_file = _write_release_files(tmp_path)

    assert load_project_version(project_file) == "1.2.3"
    assert verify_release_tag("v1.2.3", project_file, lock_file) == "v1.2.3"


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2", "v01.2.3", "v1.2.3-rc.1"])
def test_release_tag_rejects_non_release_formats(tmp_path: Path, tag: str) -> None:
    project_file, lock_file = _write_release_files(tmp_path)

    with pytest.raises(ValueError, match=r"must match vX\.Y\.Z"):
        verify_release_tag(tag, project_file, lock_file)


def test_release_tag_rejects_project_version_mismatch(tmp_path: Path) -> None:
    project_file, lock_file = _write_release_files(tmp_path)

    with pytest.raises(ValueError, match="does not match project version"):
        verify_release_tag("v1.2.2", project_file, lock_file)


def test_release_tag_rejects_lock_version_mismatch(tmp_path: Path) -> None:
    project_file, lock_file = _write_release_files(tmp_path, lock_version="1.2.2")

    with pytest.raises(ValueError, match=r"does not match uv\.lock version"):
        verify_release_tag("v1.2.3", project_file, lock_file)


def test_release_tag_cli_accepts_current_version(capsys: pytest.CaptureFixture[str]) -> None:
    tag = f"v{load_project_version(PROJECT_FILE)}"

    assert main([tag]) == 0
    assert capsys.readouterr().out == f"release tag matches project version: {tag}\n"
