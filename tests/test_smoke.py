"""Smoke tests: the package imports and the CLI reports its version."""

from typer.testing import CliRunner

from judgelab import __version__
from judgelab.cli import app

runner = CliRunner()


def test_cli_version_matches_package() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_version_is_three_part_numeric() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_bare_invocation_shows_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help exits with code 0 (Typer) or 2 depending on version;
    # either way it must surface usage rather than crash.
    assert result.exit_code in (0, 2)
    assert "Usage" in result.stdout or "Usage" in result.output
