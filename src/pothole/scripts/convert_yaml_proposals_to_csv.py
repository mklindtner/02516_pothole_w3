#!/usr/bin/env python3


from pathlib import Path


import yaml


def convert(infile, outfile):
    with Path(infile).open('rt') as file:
        proposals = yaml.safe_load(file)

    with Path(outfile).open('wt') as file:
        save_proposalls(file, proposals)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=Path)
    parser.add_argument('outfile', type=Path)
    args = parser.parse_args()

    convert(args.infile, args.outfile)


if __name__ == '__main__':
    main()
