"""Compute official All-In-One features stem-by-stem without retaining hidden FFT arrays."""
import argparse
import gc
from pathlib import Path

import numpy as np

from analyze_structure import setup_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stems-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.is_file():
        return
    setup_runtime(args.runtime_root, args.threads)
    from allin1_infer.spectrogram import build_spec_processor
    from madmom_infer.audio.signal import Signal
    processor = build_spec_processor()
    features = []
    for name in ['bass', 'drums', 'other', 'vocals']:
        samples = Signal(args.stems_dir/(name+'.wav'), num_channels=1)
        spectrogram = processor(samples)
        # ndarray subclasses retain .stft/.spectrogram owners, including full complex
        # FFT arrays. Copying to a plain ndarray preserves values and frees owners.
        features.append(np.array(spectrogram, copy=True, subok=False))
        del samples, spectrogram
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.output), np.stack(features))
    print(str(args.output), flush=True)


if __name__ == '__main__':
    main()
