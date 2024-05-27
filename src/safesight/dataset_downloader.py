import mlcroissant as mlc
import urllib.request


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


download_dataset(
    "https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage/croissant/download",
    "archive.zip",
)
