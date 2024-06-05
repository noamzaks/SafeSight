import os
import zipfile

import click
import mlcroissant as mlc
import urllib.request
import safesight.cli as cli

datasets = {
    "ckay16": "https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage/croissant/download",
}


@cli.cli.group()
def dataset():
    """Download datasets (python>=3.10)"""
    pass


@dataset.command(name="list")
def list_datasets():
    """List available datasets"""
    for name, url in datasets.items():
        click.echo(f"{name}: \t{url}")


@dataset.command()
@click.argument("name")
@click.option("-d", "--dest", default=None, help="Destination directory")
def download(name, dest):
    """Download dataset NAME into ./data/<dest>"""
    if name not in datasets:
        click.echo(f"Dataset {name} not found.")
        return
    if dest is None:
        dest = name
    download_dataset(datasets[name], f"{dest}.zip")

    with zipfile.ZipFile(f"{dest}.zip", "r") as zip_ref:
        zip_ref.extractall(f"data/{dest}")

    os.remove(f"{dest}.zip")

    click.echo(f"Done!")


def get_download_url(croissant_url: str) -> str | None:
    """
    Get the url from which to download the relevant dataset.
    Currently depends on the relevant dataset file being
    the first element in `metadata.distribution`.
    Returns `None` if it could not be found.
    Takes a url to a croissant metadata file.
    """
    ds = mlc.Dataset(croissant_url)
    try:
        download_url = ds.metadata.distribution[0].content_url
        return download_url
    except (AttributeError, IndexError):
        return None


def download_file(url: str, dest_filename: str):
    name, response = urllib.request.urlretrieve(url)
    with open(dest_filename, "wb") as file:
        with open(name, "rb") as local_file:
            file.write(local_file.read())


def download_dataset(croissant_url: str, dest_filename: str):
    download_url = get_download_url(croissant_url)
    if download_url:
        download_file(download_url, dest_filename)
