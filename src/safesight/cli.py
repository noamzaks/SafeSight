import sys

import click


def is_venv():
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)


def is_python_exactly(version):
    return sys.version.startswith(version)


def is_python_at_least(version):
    return sys.version_info >= version


@click.group()
def cli():
    if not is_venv():
        click.echo("You are not in a virtual environment.")
        sys.exit(1)


if is_python_at_least((3, 10)):
    # noinspection PyUnresolvedReferences
    import safesight.dataset_downloader
else:
    @cli.command()
    @click.argument("_", nargs=-1)
    def dataset(_):
        """Download datasets (python>=3.10)"""
        click.echo("This command requires Python 3.10 or later.")
        sys.exit(1)


def main():
    cli()
