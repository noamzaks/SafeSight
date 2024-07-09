from typing import List
from tqdm import tqdm

import click
import pytube
import pytube.exceptions
from pytube import Search
from pytube.innertube import _default_clients
from safesight.cli import youtube_downloader


def search(search_term: str, video_count: int) -> List[pytube.YouTube]:
    """
    Searches for `search_term` and returns a list of the first `num_videos` that result.
    """

    search = Search(search_term)
    if not search.results:
        return []
    while len(search.results) < video_count:
        search.get_next_results()
    return search.results[:video_count]


@youtube_downloader.command()
@click.option("--search-term", type=str, help="String to search for on youtube")
@click.option(
    "--output-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to download videos to",
)
@click.option(
    "--video-count",
    type=int,
    default=10,
    help="Number of videos to download",
    show_default=True,
)
def download(search_term: str, output_path: str, video_count: int):
    """
    Searches for and downloads the first `num_videos` that come up when searching for
    the provided term, at the lowest resolution possible.
    """
    # To bypass age-restriction: (see https://stackoverflow.com/questions/75791765/how-to-download-videos-that-require-age-verification-with-pytube)
    _default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

    results = search(search_term, video_count)

    for video in tqdm(results):
        try:
            stream = video.streams.get_lowest_resolution()

            if stream:
                stream.download(output_path, timeout=10)
        except pytube.exceptions.PytubeError:
            pass
