from pathlib import Path

import pytube
from pytube import Search
from pytube.innertube import _default_clients


def search_and_download(search_term: str, output_path: str):
    """
    Searches for and donwloads the first set of videos (about 20) that come up when searching for
    the provided term.
    """
    search = Search(search_term)
    assert search.results is not None

    for video in search.results:
        assert type(video) is pytube.YouTube

        stream = video.streams.get_lowest_resolution()
        assert stream is not None

        stream.download(output_path)


def main():
    # To bypass age-restriction: (see https://stackoverflow.com/questions/75791765/how-to-download-videos-that-require-age-verification-with-pytube)
    _default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]

    output_path = Path("./youtube_dataset/")
    output_path.mkdir(parents=True, exist_ok=True)
    search_and_download("traffic accident", str(output_path))


if __name__ == "__main__":
    main()
