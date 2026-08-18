"""Command-line interface for judgelab."""

from __future__ import annotations

import typer

from judgelab import __version__

app = typer.Typer(
    name="judgelab",
    help="Audit the reliability of LLM-as-a-judge systems.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Audit the reliability of LLM-as-a-judge systems.

    A callback is defined so Typer treats this as a multi-command group (rather
    than a single-command app), which is required for subcommands to keep their
    name once ``run``/``report``/``compare`` are added.
    """


@app.command()
def version() -> None:
    """Print the installed judgelab version."""
    typer.echo(__version__)


@app.command()
def report(
    *,
    check: bool = typer.Option(
        False,
        "--check",
        help="Verify the committed report matches a fresh recompute; exit 1 on drift.",
    ),
) -> None:
    """Regenerate (or, with --check, verify) the MT-Bench agreement report."""
    from judgelab.report import JSON_NAME, MD_NAME, check_report, write_report

    if check:
        if check_report():
            typer.echo("report up to date")
        else:
            typer.echo("report DRIFT: committed report differs from a fresh recompute", err=True)
            raise typer.Exit(code=1)
    else:
        write_report()
        typer.echo(f"wrote reports/{MD_NAME} and reports/{JSON_NAME}")


if __name__ == "__main__":
    app()
