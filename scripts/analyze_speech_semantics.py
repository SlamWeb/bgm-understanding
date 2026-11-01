"""Offline SenseVoice INT8 speech transcript and vocal-tone emotion candidates.

Use on speech-dominant audio. Emotion tags describe vocal delivery, not musical
mood, lyrical meaning, or the speaker's true feelings. Token timestamps are
model onset estimates; token ends are explicitly estimated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


MODEL_SOURCE = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_audio(path):
    rate, samples = wavfile.read(path)
    if np.issubdtype(samples.dtype, np.unsignedinteger):
        span = (np.iinfo(samples.dtype).max + 1) / 2
        samples = (samples.astype(np.float32) - span) / span
    elif np.issubdtype(samples.dtype, np.signedinteger):
        samples = samples.astype(np.float32) / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    else:
        samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Expected finite, nonempty PCM audio.')
    duration = len(samples) / rate
    if rate != 16000:
        common = math.gcd(int(rate), 16000)
        samples = resample_poly(samples, 16000 // common, int(rate) // common).astype(np.float32)
    return samples, duration


def native_path(path):
    """Prefer an existing Windows short path for C++ runtimes with narrow I/O."""
    value = str(Path(path).resolve())
    if os.name == 'nt':
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(value, buffer, len(buffer)):
            return buffer.value
    return value


def speech_chunks(samples, rate=16000, max_seconds=20, pause_seconds=.5):
    """Energy pauses bound memory and context; they are not a learned speech VAD."""
    hop = int(rate * .02)
    levels = np.asarray([np.sqrt(np.mean(samples[i:i + hop].astype(np.float64) ** 2))
                         for i in range(0, len(samples), hop)])
    if not len(levels) or float(levels.max()) < 1e-5:
        return []
    threshold = max(1e-4, float(np.percentile(levels, 90)) * .025)
    active = np.flatnonzero(levels >= threshold)
    if not len(active):
        return []
    rows, first, previous = [], int(active[0]), int(active[0])
    for current in map(int, active[1:]):
        if (current - previous) * hop / rate > pause_seconds:
            rows.append((max(0, first * hop - int(.12 * rate)), min(len(samples), (previous + 1) * hop + int(.12 * rate))))
            first = current
        previous = current
    rows.append((max(0, first * hop - int(.12 * rate)), min(len(samples), (previous + 1) * hop + int(.12 * rate))))
    output = []
    limit = int(max_seconds * rate)
    for start, end in rows:
        if end - start < int(.2 * rate):
            continue
        for offset in range(start, end, limit):
            stop = min(end, offset + limit)
            if stop > offset:
                output.append((offset, stop))
    return output


def clean_tag(value):
    return str(value or '').replace('<|', '').replace('|>', '').strip()


def split_tokens(tokens, timestamps, offset, chunk_end, gap_seconds=.5):
    """Preserve acoustic token starts, estimate ends, group on punctuation/pauses."""
    if len(tokens) != len(timestamps):
        raise ValueError('SenseVoice tokens/timestamps have different lengths.')
    rows = []
    for i, (token, local) in enumerate(zip(tokens, timestamps)):
        if not isinstance(local, (int, float)) or not math.isfinite(local):
            raise ValueError('Nonfinite SenseVoice token timestamp.')
        start = offset + float(local)
        if not 0 <= start <= chunk_end + .05:
            raise ValueError('SenseVoice token timestamp outside decoded chunk.')
        if re.fullmatch(r'<\|.*?\|>', token):
            continue
        next_start = offset + float(timestamps[i + 1]) if i + 1 < len(tokens) else chunk_end
        if next_start < start - .001:
            raise ValueError('SenseVoice timestamps are not monotonic.')
        end = min(chunk_end, max(start + .02, min(next_start, start + .3)))
        if end <= start:
            continue
        rows.append({'text': str(token).replace('▁', ' '), 'start': round(start, 4), 'end': round(end, 4),
                     'start_method': 'SenseVoice CTC token timestamp',
                     'end_method': 'estimated from next token onset, capped at 300ms'})
    groups, group = [], []
    for row in rows:
        if group and row['start'] - group[-1]['start'] > gap_seconds:
            groups.append(group)
            group = []
        group.append(row)
        if re.search(r'[。.!！?？;；]$', row['text'].strip()):
            groups.append(group)
            group = []
    if group:
        groups.append(group)
    result = []
    for words in groups:
        text = ''.join(w['text'] for w in words).strip()
        if text:
            result.append({'start': words[0]['start'], 'end': words[-1]['end'], 'text': text,
                           'type': 'speech_candidate', 'words': words, 'needs_review': True,
                           'timing_status': 'model token onset estimates with estimated ends'})
    return result


def analyze(wav, output, root, threads, start_seconds=0, selected_duration=None, pause_seconds=.8):
    started = time.monotonic()
    folder = root / 'models/sensevoice-int8'
    model, tokens = folder / 'model.int8.onnx', folder / 'tokens.txt'
    if not model.is_file() or not tokens.is_file():
        raise FileNotFoundError('SenseVoice INT8 model/tokens are not deployed under runtime-root/models/sensevoice-int8.')
    # sherpa-onnx-core ships its matched ORT DLL separately. Pin this local DLL
    # before importing the extension; another audio runtime may use an older ORT.
    dll_handle = None
    native = root / 'speech-packages/native'
    if os.name == 'nt' and native.is_dir():
        dll_handle = os.add_dll_directory(str(native.resolve()))
        os.environ['PATH'] = str(native.resolve()) + os.pathsep + os.environ.get('PATH', '')
        import ctypes
        ctypes.WinDLL(str(native / 'onnxruntime.dll'))
    sys.path.insert(0, str(root / 'speech-packages'))
    import sherpa_onnx
    samples, duration = load_audio(wav)
    if start_seconds >= duration:
        raise ValueError('Selected start is outside audio.')
    selected_end = min(duration, start_seconds + selected_duration) if selected_duration is not None else duration
    selection_start = int(round(start_seconds * 16000))
    selection_end = min(len(samples), int(round(selected_end * 16000)))
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=native_path(model), tokens=native_path(tokens), num_threads=threads,
        sample_rate=16000, feature_dim=80, provider='cpu', language='auto', use_itn=True,
        debug=False)
    windows = [(first + selection_start, last + selection_start)
               for first, last in speech_chunks(samples[selection_start:selection_end], pause_seconds=pause_seconds)]
    segments, emotions, raw = [], [], []
    for first, last in windows:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples[first:last])
        recognizer.decode_stream(stream)
        result = stream.result
        record = {key: getattr(result, key) for key in ('text', 'lang', 'emotion', 'event')}
        record.update({key: list(getattr(result, key))
                       for key in ('tokens', 'timestamps', 'durations', 'ys_log_probs')})
        start, end = first / 16000, min(duration, last / 16000)
        raw.append({'start': start, 'end': end, 'result': record})
        parsed = split_tokens(record.get('tokens', []), record.get('timestamps', []), start, end)
        if not parsed and record.get('text', '').strip():
            parsed = [{'start': start, 'end': end, 'text': record['text'].strip(),
                       'type': 'speech_candidate', 'words': [], 'needs_review': True,
                       'timing_status': 'energy-bounded acoustic decoding chunk, not word-aligned'}]
        for segment in parsed:
            segment.update(language_candidate=clean_tag(record.get('lang')), emotion_candidate=clean_tag(record.get('emotion')),
                           event_candidate=clean_tag(record.get('event')))
        segments.extend(parsed)
        emotions.append({'start': round(start, 4), 'end': round(end, 4),
                         'label_candidate': clean_tag(record.get('emotion')) or 'unknown',
                         'event_candidate': clean_tag(record.get('event')) or 'unknown',
                         'language_candidate': clean_tag(record.get('lang')) or 'unknown',
                         'score': None, 'score_meaning': 'Native SenseVoice tag; this API exposes no calibrated tag probability.'})
    result = {
        'schema': 'bgm-speech-semantics-v1',
        'audio': {'source': str(wav.resolve()), 'sha256': digest(wav), 'duration_seconds': duration,
                  'time_origin': 'canonical_wav_start'},
        'speech_emotion': {'status': 'model_candidates' if emotions else 'no_speech_detected',
                           'method': 'SenseVoiceSmall INT8 native vocal emotion tags', 'timeline': emotions,
                           'scope': 'Vocal delivery emotion candidate, not music mood, lyric sentiment, or speaker intent.',
                           'coverage': 'Energy-active decoding chunks only; unlisted times are not emotion predictions.'},
        'secondary_transcript': {'status': 'complete' if segments else 'no_supported_transcript',
                                 'method': 'SenseVoiceSmall INT8 acoustic CTC decoding', 'segments': segments,
                                 'full_text': ' '.join(r['result'].get('text', '') for r in raw).strip(),
                                 'human_reviewed': False,
                                 'timing_note': 'Model token starts; token ends estimated at next onset or 300ms. Segment ends inherit these estimates.',
                                 'role': 'Independent acoustic second opinion for speech; never silently replaces another ASR report.'},
        'coverage': {'selection_start_seconds': start_seconds, 'selection_end_seconds': selected_end,
                     'decoded_chunks': [{'start': first / 16000, 'end': min(duration, last / 16000)} for first, last in windows],
                     'pause_split_seconds': pause_seconds, 'time_origin': 'canonical_wav_start'},
        'raw_chunks': raw,
        'model': {'name': 'SenseVoiceSmall 2024-07-17 INT8', 'source': MODEL_SOURCE,
                  'weights': str(model), 'sha256': digest(model), 'bytes': model.stat().st_size,
                  'tokens_sha256': digest(tokens), 'runtime_version': sherpa_onnx.__version__,
                  'device': 'cpu', 'threads': threads},
        'elapsed_seconds': round(time.monotonic() - started, 3),
        'human_reviewed': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(output), 'segments': len(segments),
                      'emotion_candidates': len(emotions), 'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wav', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, default=Path('D:/SelfMedia/管理/工具/bgm-analysis'))
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--duration', type=float)
    p.add_argument('--pause-seconds', type=float, default=.8)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists; use a fresh result file.')
    if not 1 <= args.threads <= 4:
        p.error('threads must be in [1,4].')
    if not math.isfinite(args.start) or args.start < 0 or (args.duration is not None and
            (not math.isfinite(args.duration) or args.duration <= 0)) or not math.isfinite(args.pause_seconds) or args.pause_seconds < .1:
        p.error('Require nonnegative start, positive duration, and pause-seconds >= .1.')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    analyze(args.wav, args.output, args.runtime_root, args.threads, args.start, args.duration, args.pause_seconds)


if __name__ == '__main__':
    main()
