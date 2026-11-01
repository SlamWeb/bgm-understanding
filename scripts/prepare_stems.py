"""Run source separation in an isolated process so its memory is freed before structure inference."""
import argparse
from pathlib import Path

from analyze_structure import setup_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--demix-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    setup_runtime(args.runtime_root, args.threads)
    from allin1_infer.demix import demix
    demix([args.wav], args.demix_dir, 'cpu')


if __name__ == '__main__':
    main()
