"""Keyless drift gate: the committed report must match a fresh recompute."""

import json

from typer.testing import CliRunner

from judgelab.cli import app
from judgelab.report import JSON_NAME, MD_NAME, reports_dir

runner = CliRunner()


def test_report_files_present() -> None:
    directory = reports_dir()
    assert (directory / MD_NAME).exists()
    assert (directory / JSON_NAME).exists()


def test_cli_report_check_passes() -> None:
    # Recomputes from the committed snapshot and byte-compares to the committed report.
    result = runner.invoke(app, ["report", "--check"])
    assert result.exit_code == 0, result.output
    assert "up to date" in result.output


def test_committed_json_headline_numbers() -> None:
    data = json.loads((reports_dir() / JSON_NAME).read_text(encoding="utf-8"))
    assert data["n_comparisons"] == 1814
    assert data["ties_excluded"]["raw_agreement"] == 0.884
    assert data["ties_excluded"]["cohen_kappa"] == 0.767
    assert data["with_ties"]["cohen_kappa"] == 0.5047
    assert data["position_inconsistency_rate"] == 0.1599
    assert data["human_krippendorff_alpha"] == 0.4855  # human-human agreement ceiling
    assert data["n_human_multi_rated"] == 961
