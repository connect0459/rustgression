import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="version scripts are POSIX shell scripts, not applicable on Windows",
)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_VERSION_UTILS = _PROJECT_ROOT / "scripts" / "libs" / "version-utils.sh"


def _extract_section_versions(toml_text: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    current_section: str | None = None
    for line in toml_text.splitlines():
        if line.startswith("["):
            current_section = line
        elif (
            line.startswith('version = "')
            and current_section is not None
            and current_section not in versions
        ):
            versions[current_section] = line.split('"')[1]
    return versions


def _apply_version_write(pyproject_path: Path, new_version: str) -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {shlex.quote(str(_VERSION_UTILS))} && "
            f'awk -v ver={shlex.quote(new_version)} "$_AWK_PYPROJECT_WRITE_VERSION" '
            f"{shlex.quote(str(pyproject_path))}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    pyproject_path.write_text(result.stdout)


def _read_project_version(pyproject_path: Path) -> str:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {shlex.quote(str(_VERSION_UTILS))} && "
            f'awk "$_AWK_PYPROJECT_READ_VERSION" {shlex.quote(str(pyproject_path))} '
            "| cut -d '\"' -f2",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def pyproject_with_tool_version(tmp_path: Path) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text(
        textwrap.dedent("""\
            [project]
            name = "testpkg"
            version = "0.1.0"

            [tool.some-tool]
            version = "1.0.0"
        """)
    )
    return p


def test_version_write_leaves_tool_section_versions_unchanged(
    pyproject_with_tool_version: Path,
) -> None:
    """Only the [project] version field is updated; [tool.*] entries are preserved."""
    _apply_version_write(pyproject_with_tool_version, "0.2.0")
    versions = _extract_section_versions(pyproject_with_tool_version.read_text())
    assert versions["[project]"] == "0.2.0"
    assert versions["[tool.some-tool]"] == "1.0.0"


def test_version_write_preserves_all_tool_section_versions(tmp_path: Path) -> None:
    """All [tool.*] version entries survive an update when multiple tool sections are present."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent("""\
            [project]
            name = "testpkg"
            version = "0.1.0"

            [tool.ruff]
            version = "2.0.0"

            [tool.mypy]
            version = "3.0.0"
        """)
    )
    _apply_version_write(pyproject, "0.2.0")
    versions = _extract_section_versions(pyproject.read_text())
    assert versions["[project]"] == "0.2.0"
    assert versions["[tool.ruff]"] == "2.0.0"
    assert versions["[tool.mypy]"] == "3.0.0"


def test_version_read_returns_project_section_value_only(
    pyproject_with_tool_version: Path,
) -> None:
    """Version extraction returns only the [project] value, ignoring [tool.*] entries."""
    assert _read_project_version(pyproject_with_tool_version) == "0.1.0"
