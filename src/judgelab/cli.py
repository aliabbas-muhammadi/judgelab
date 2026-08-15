"""Command-line interface for judgelab.

Subcommands (evaluation, reporting, probes) land in later contributions; this
module currently exposes only ``version`` so the console script is wired and
smoke-tested from day one.
"""

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


if __name__ == "__main__":
    app()
