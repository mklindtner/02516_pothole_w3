#!/usr/bin/env python3


from pathlib import Path


import yaml


from pothole.boxes import save_proposals


def convert(infile, outfile):
    with Path(infile).open('rt') as file:
        proposals = yaml.safe_load(file)

    with Path(outfile).open('xt') as file:
        save_proposals(file, proposals)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=Path)
    parser.add_argument('outfile', type=Path)
    args = parser.parse_args()

    # Fail early.
    if args.outfile.is_file():
        raise RuntimeError(f'File "{args.outfile}" already exists!')

    convert(args.infile, args.outfile)


if __name__ == '__main__':
    main()
