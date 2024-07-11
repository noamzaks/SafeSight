import importlib
import sys
from typing import List

import click


class ModuleGroup(click.Group):
    def __init__(
        self,
        *args,
        module: str = None,
        python_accepted_versions: List[str] = None,
        **kwargs,
    ):
        if python_accepted_versions is not None:
            for version in python_accepted_versions:
                if sys.version.startswith(version):
                    break
            else:
                kwargs["help"] = (
                    "[Unavailable] Supported Python versions: "
                    + ", ".join(python_accepted_versions)
                )
                module = None
        super().__init__(*args, **kwargs)
        self.module = None
        self.module_name = module

    def list_commands(self, ctx):
        if self.module is None:
            if self.module_name is None:
                return []
            self.module = importlib.import_module(self.module_name)
        return super().list_commands(ctx)

    def get_command(self, ctx, cmd_name):
        if self.module is None:
            if self.module_name is None:
                return click.Command(
                    cmd_name,
                    callback=lambda: click.echo(
                        "This command is not available at current python version."
                    ),
                )
            self.module = importlib.import_module(self.module_name)
        return super().get_command(ctx, cmd_name)


def is_venv():
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


@click.group()
def cli():
    if not is_venv():
        click.echo("You are not in a virtual environment.")
        sys.exit(1)


@cli.group(
    cls=ModuleGroup,
    module="safesight.dataset_downloader",
    python_accepted_versions=["3.10", "3.11", "3.12"],
)
def dataset():
    """Download datasets (python>=3.10)"""
    pass


@cli.group(
    cls=ModuleGroup,
    module="safesight.test_gemini",
    python_accepted_versions=["3.9", "3.10", "3.11", "3.12"],
)
def gemini():
    """Commands for Gemini (python>=3.9)"""
    pass


@cli.group(
    cls=ModuleGroup, module="safesight.test_blip", python_accepted_versions=["3.8"]
)
def lavis():
    """Commands for the LAVIS library (BLIP model) (python==3.8)"""
    pass


@cli.group(
    cls=ModuleGroup, module="safesight.test_yolo", python_accepted_versions=["3.8"]
)
def yolo():
    """Commands for YOLO model"""
    pass


@cli.group(
    cls=ModuleGroup, module="safesight.analyzer", python_accepted_versions=["3.8"]
)
def analyzer():
    """Commands for running the Analyzer"""
    pass


@cli.group(cls=ModuleGroup, module="safesight.nvidia")
def nvidia():
    """
    Commands for interacting with the NVIDIA API.
    """


@cli.group(cls=ModuleGroup, module="safesight.run_pipeline")
def pipeline():
    """
    Run pipeline on dataset.
    """


@cli.group(cls=ModuleGroup, module="safesight.youtube_downloader")
def youtube_downloader():
    """
    Commands for searching and downloading videos from youtube.
    """


def main():
    cli()
