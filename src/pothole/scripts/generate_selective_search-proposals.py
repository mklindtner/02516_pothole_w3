#!/usr/bin/env python3


import logging
from pathlib import Path


import yaml


from pothole.boxes import filter_proposals, xyxy_to_xywh
from pothole.boxes.selective_search import run_selective_search
from pothole.datasets import DEFAULT_BASE_PATH, PotholeRawData


LOG = logging.getLogger()


DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / 'data'


def generate_proposals(raw_data, k1, k2):
    results = {}

    for xml, img, target in raw_data.iter_subset_image_boxes('train'):
        proposals = run_selective_search(img)

        pos, bg = filter_proposals(proposals, target, k1=k1, k2=k2)

        LOG.info(
            '%s: pos=%d bg=%d (%d total props, %d ground truth)',
            xml,
            len(pos),
            len(bg),
            len(proposals),
            len(target),
        )

        results[str(xml)] = {
            'pothole': list(map(lambda b: b.tolist(), pos)),
            'background': list(map(lambda b: b.tolist(), bg)),
        }

        break

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=DEFAULT_BASE_PATH, type=Path)
    parser.add_argument('--k1', default=0.3, type=float)
    parser.add_argument('--k2', default=0.7, type=float)
    parser.add_argument('--output-prefix', default='props-%(k1)s-%(k2)s')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, type=Path)
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

    stem = args.output_prefix % {
        'k1': f'{args.k1:.2f}',
        'k2': f'{args.k2:.2f}',
    }
    output_path = args.output_dir / f'{stem}.yml'
    if output_path.is_file():
        raise RuntimeError(f'File "{output_path}" already exists!.')

    raw_data = PotholeRawData(base_path=args.data_path)

    LOG.info(
        'Generating proposals, filtering with k1=%.3f and k2=%.3f and saving '
        'into "%s".',
        args.k1,
        args.k2,
        output_path,
    )

    result = generate_proposals(raw_data, args.k1, args.k2)

    with output_path.open('xt') as file:
        yaml.safe_dump(result, file, default_flow_style=False)


if __name__ == '__main__':
    main()
