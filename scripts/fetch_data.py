#!/usr/bin/env python3


import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request
from zipfile import ZipFile


LOG = logging.getLogger()


def fetch_data(url, output_path):
    with TemporaryDirectory() as dir:
        zippath = (Path(dir) / 'Potholes.zip')

        with zippath.open('wb') as file:
            with urllib.request.urlopen(url) as response:
                for chunk in iter(lambda: response.read(4096), b''):
                    file.write(chunk)

        with ZipFile(zippath) as zipfile:
            zipfile.extractall(path=output_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url',
        default='https://drive.usercontent.google.com/download?id=1N39KSmiQXPGBKy_twSomTl64mP-8TdYn&export=download&confirm=t&uuid=6d3ebe7d-e32d-4744-9a1b-2029a722f81c',
    )
    parser.add_argument(
        '--output-path',
        default=Path(__file__).parent.parent / 'data',
    )
    parser.add_argument('--debug', '-v', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARN
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format='%(message)s')

    args.output_path.mkdir(parents=True, exist_ok=True)

    LOG.info(
        'Fetching data from "%s" and saving them into "%s".',
        args.url,
        args.output_path,
    )
    fetch_data(args.url, args.output_path)


if __name__ == '__main__':
    main()
