"""CPU music sound tags and mood candidates; bounded inference, no downloads at run time."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


MOODS = ('happy', 'sad', 'relaxed', 'aggressive')
SPEECH = ('Speech', 'Child speech, kid speaking', 'Conversation', 'Narration, monologue',
          'Speech synthesizer', 'Whispering')
SINGING = ('Singing', 'Choir', 'Yodeling', 'Chant', 'Mantra', 'Child singing',
           'Synthetic singing', 'Rapping', 'Humming', 'Vocal music', 'A capella')
INSTRUMENTS = ('Guitar', 'Electric guitar', 'Bass guitar', 'Acoustic guitar', 'Steel guitar, slide guitar',
               'Banjo', 'Sitar', 'Mandolin', 'Zither', 'Ukulele', 'Piano', 'Electric piano', 'Organ',
               'Electronic organ', 'Hammond organ', 'Synthesizer', 'Sampler', 'Harpsichord', 'Percussion',
               'Drum kit', 'Drum machine', 'Drum', 'Snare drum', 'Rimshot', 'Bass drum', 'Timpani',
               'Tabla', 'Cymbal', 'Hi-hat', 'Tambourine', 'Maraca', 'Gong', 'Tubular bells',
               'Marimba, xylophone', 'Glockenspiel', 'Vibraphone', 'Steelpan', 'Orchestra',
               'Brass instrument', 'French horn', 'Trumpet', 'Trombone', 'Bowed string instrument',
               'String section', 'Violin, fiddle', 'Pizzicato', 'Cello', 'Double bass', 'Flute',
               'Saxophone', 'Clarinet', 'Harp', 'Bell', 'Church bell', 'Chime', 'Wind chime',
               'Harmonica', 'Accordion', 'Bagpipes', 'Didgeridoo', 'Theremin', 'Singing bowl')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_wav(path):
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
        raise ValueError('Expected finite, nonempty WAV audio.')
    duration = len(samples) / rate
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        samples = resample_poly(samples, 16000 // divisor, int(rate) // divisor).astype(np.float32)
    return samples, duration


def mel_filter(nfft, n_mels, low, high, slaney=False):
    frequencies = np.linspace(0, 8000, nfft // 2 + 1)
    if slaney:
        def to_mel(f):
            f = np.asarray(f, dtype=np.float64)
            return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1) / 1000) / (np.log(6.4) / 27), f / (200 / 3))
        def to_hz(m):
            return np.where(m >= 15, 1000 * np.exp((m - 15) * (np.log(6.4) / 27)), m * (200 / 3))
        edges = to_hz(np.linspace(to_mel(low), to_mel(high), n_mels + 2))
        rows = [np.maximum(0, np.minimum((frequencies - a) / (b - a), (c - frequencies) / (c - b)))
                * (2 / (c - a)) for a, b, c in zip(edges, edges[1:], edges[2:])]
    else:
        warped = 1127 * np.log1p(frequencies / 700)
        edges = np.linspace(1127 * np.log1p(low / 700), 1127 * np.log1p(high / 700), n_mels + 2)
        rows = [np.maximum(0, np.minimum((warped - a) / (b - a), (c - warped) / (c - b)))
                for a, b, c in zip(edges, edges[1:], edges[2:])]
    weights = np.asarray(rows, dtype=np.float32).T
    weights[0] = 0
    return weights


def yamnet_patches(samples, starts):
    """Google YAMNet frontend equations, no centered STFT or power-spectrum mismatch."""
    filt = mel_filter(512, 64, 125, 7500)
    hann = (.5 - .5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    patches = []
    for start in starts:
        segment = samples[start:start + 15600]
        segment = np.pad(segment, (0, max(0, 15600 - len(segment))))
        frames = np.lib.stride_tricks.sliding_window_view(segment, 400)[::160][:96]
        magnitude = np.abs(np.fft.rfft(frames * hann, n=512, axis=1)).astype(np.float32)
        patches.append(np.log(magnitude @ filt + .001))
    return np.asarray(patches, dtype=np.float32)[:, None, :, :]


def yamnet_patch_starts(sample_count):
    extra_hops = int(math.ceil(max(0, sample_count - 15600) / 7680))
    return [hop * 7680 for hop in range(extra_hops + 1)]


def group_activity(timeline):
    groups = []
    for window in timeline:
        kind = window['activity']
        if kind not in ('speech_candidate', 'singing_candidate', 'mixed_vocal_candidate', 'vocal_candidate', 'uncertain_vocal_candidate'):
            continue
        if groups and groups[-1]['type'] == kind and abs(groups[-1]['end'] - window['start']) < .001:
            groups[-1]['end'] = window['end']
            groups[-1]['peak_score'] = max(groups[-1]['peak_score'], window['vocal_score'])
        else:
            groups.append({'start': window['start'], 'end': window['end'], 'type': kind,
                           'peak_score': window['vocal_score']})
    return groups


def sound_tags(samples, duration, root, threads):
    import torch
    sys.path.insert(0, str(root / 'panns-test/yamnet-source'))
    from torch_audioset.yamnet.model import YAMNet, yamnet_category_metadata
    torch.set_num_threads(threads)
    model = YAMNet()
    weights = root / 'panns-test/yamnet.pth'
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.eval()
    labels = [v['name'] for v in yamnet_category_metadata()]
    speech_ids = [labels.index(v) for v in SPEECH if v in labels]
    singing_ids = [labels.index(v) for v in SINGING if v in labels]
    inst_ids = [labels.index(v) for v in INSTRUMENTS if v in labels]
    # Follow Google's pad_waveform: first receptive field is 15600 samples;
    # only add another 7680-sample hop when real audio extends past it.
    # A cell-per-hop loop would add an almost fully padded tail classification.
    starts = yamnet_patch_starts(len(samples))
    probabilities = []
    with torch.inference_mode():
        for first in range(0, len(starts), 4):
            tensor = torch.from_numpy(yamnet_patches(samples, starts[first:first + 4]))
            score = model(tensor, to_prob=True).numpy()
            probabilities.append(score)
    scores = np.concatenate(probabilities)
    vocal_windows, instrument_windows = [], []
    for index, row in enumerate(scores):
        start = starts[index] / 16000
        end = starts[index + 1] / 16000 if index + 1 < len(starts) else duration
        if end <= start:
            continue
        speech, singing = float(max(row[speech_ids])), float(max(row[singing_ids]))
        vocal = max(speech, singing)
        if speech >= .3 and singing >= .3:
            activity = 'mixed_vocal_candidate'
        elif speech >= .3:
            activity = 'speech_candidate'
        elif singing >= .3:
            activity = 'singing_candidate'
        else:
            activity = 'uncertain' if vocal >= .1 else 'no_vocal_evidence'
        vocal_windows.append({'start': round(start, 6), 'end': round(end, 6),
                              'speech_score': round(speech, 5), 'singing_score': round(singing, 5),
                              'vocal_score': round(vocal, 5), 'activity': activity})
    # Aggregate neighbouring tags to reduce flicker and downstream context size.
    for first in range(0, len(scores), 8):
        chunk = scores[first:first + 8]
        means = chunk.mean(axis=0)
        selected = sorted(inst_ids, key=lambda i: float(means[i]), reverse=True)[:6]
        instrument_windows.append({'start': round(starts[first] / 16000, 6),
                                   'end': round(starts[first + 8] / 16000 if first + 8 < len(starts) else duration, 6),
                                   'candidates': [{'label': labels[i], 'score': round(float(means[i]), 5),
                                                   'peak_score': round(float(chunk[:, i].max()), 5)}
                                                  for i in selected if means[i] >= .025]})
    means = scores.mean(axis=0)
    top_ids = sorted(inst_ids, key=lambda i: float(means[i]), reverse=True)[:10]
    summary = [{'label': labels[i], 'mean_score': round(float(means[i]), 5),
                'peak_score': round(float(scores[:, i].max()), 5)} for i in top_ids if means[i] >= .015]
    del model, scores, probabilities
    gc.collect()
    return ({'status': 'model_candidates', 'method': 'YAMNet AudioSet window classification',
             'summary': summary, 'timeline': instrument_windows,
             'score_meaning': 'Uncalibrated independent class scores, not confirmed instrument identity or source onset.',
             'temporal_resolution_seconds': 3.84},
            {'status': 'model_candidates', 'method': 'YAMNet explicit speech and singing classes',
             'timeline': vocal_windows, 'segments': group_activity(vocal_windows),
             'activity_threshold': .3, 'uncertain_threshold': .1,
             'score_meaning': 'Maximum score among explicit speech/singing classes; not calibrated vocal-presence probability.',
             'hop_seconds': .48, 'receptive_field_seconds': .975,
             'timing_note': 'Cells partition the audio; predictions use up to 0.975s of context starting at cell start. Boundaries are approximate.'},
            {'name': 'YAMNet', 'weights': str(weights), 'sha256': digest(weights), 'bytes': weights.stat().st_size,
             'source': 'https://github.com/w-hc/torch_audioset',
             'official_frontend_reference': 'https://github.com/tensorflow/models/tree/master/research/audioset/yamnet',
             'implementation': 'Existing third-party PyTorch weight conversion; frontend reimplemented from Google equations in NumPy.',
             'frontend_reference_check': {'reference': 'Google TensorFlow Models VGGish NumPy mel_features, YAMNet parameters',
                                          'logmel_max_absolute_error': 5.433129128817882e-7,
                                          'scope': 'Feature equations checked; complete neural conversion remains unverified.'},
             'numerical_equivalence_to_official_runtime': 'not_verified', 'device': 'cpu'})


def musicnn_spectrogram(samples):
    """NumPy implementation of the published Essentia TensorflowInputMusiCNN parameters."""
    padded = np.pad(samples, (256, 512))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 512)[::256]
    # Essentia's default Windowing Hann is symmetric, unnormalized for this frontend.
    window = np.hanning(512).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    weights = mel_filter(512, 96, 0, 8000, slaney=True)
    return np.log10(1 + 10000 * ((spectrum * spectrum) @ weights)).astype(np.float32)


def moods(samples, duration, root, threads):
    sys.path.insert(0, str(root / 'semantic-packages'))
    import onnxruntime as ort
    folder = root / 'models/semantic-heads'
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = False
    backbone_path = folder / 'discogs-effnet-bsdynamic-1.onnx'
    backbone = ort.InferenceSession(str(backbone_path), sess_options=opts, providers=['CPUExecutionProvider'])
    spec = musicnn_spectrogram(samples)
    # Whole-file coverage with repeated last frame, documented in output.
    frame_starts = list(range(0, len(spec), 128))
    embeddings = []
    for first in frame_starts:
        patch = spec[first:first + 128]
        if len(patch) < 128:
            patch = np.pad(patch, ((0, 128 - len(patch)), (0, 0)), mode='edge')
        embedding_name = next(v.name for v in backbone.get_outputs()
                              if v.name == 'embeddings' or v.shape[-1] == 1280)
        embeddings.append(backbone.run([embedding_name], {backbone.get_inputs()[0].name: patch[None]})[0][0])
    embedding = np.asarray(embeddings, dtype=np.float32)
    del backbone, spec, embeddings
    gc.collect()
    all_scores, provenance = {}, []
    voice_path = folder / 'voice_instrumental-discogs-effnet-1.onnx'
    voice_meta = json.loads(voice_path.with_suffix('.json').read_text(encoding='utf-8'))
    voice_head = ort.InferenceSession(str(voice_path), sess_options=opts, providers=['CPUExecutionProvider'])
    voice_input = embedding
    if len(voice_head.get_inputs()[0].shape) == 4:
        voice_input = voice_input[:, None, None, :]
    voice_name = next(v.name for v in voice_head.get_outputs() if v.shape[-1] == 2)
    voice_scores = np.asarray(voice_head.run([voice_name], {voice_head.get_inputs()[0].name: voice_input})[0]).reshape(len(embedding), 2)
    voice_index = voice_meta['classes'].index('voice')
    music_voice = [{'start': round(frame_starts[i] * .016, 6),
                    'end': round(min(duration, (frame_starts[i] + 128) * .016), 6),
                    'voice_score': round(float(voice_scores[i, voice_index]), 5)}
                   for i in range(len(embedding)) if frame_starts[i] * .016 < duration]
    del voice_head
    provenance.append({'name': voice_meta['name'], 'weights': str(voice_path), 'sha256': digest(voice_path),
                       'bytes': voice_path.stat().st_size, 'official_source': voice_meta['link'],
                       'download_source': 'https://huggingface.co/rlupi/deep-cuts-models-noncommercial',
                       'license': 'CC BY-NC-SA 4.0', 'device': 'cpu'})
    instrument_path = folder / 'mtg_jamendo_instrument-discogs-effnet-1.onnx'
    instrument_meta = json.loads(instrument_path.with_suffix('.json').read_text(encoding='utf-8'))
    instrument_head = ort.InferenceSession(str(instrument_path), sess_options=opts, providers=['CPUExecutionProvider'])
    inst_input = embedding
    if len(instrument_head.get_inputs()[0].shape) == 4:
        inst_input = inst_input[:, None, None, :]
    inst_output_name = next(v.name for v in instrument_head.get_outputs() if v.shape[-1] == 40)
    inst_scores = instrument_head.run([inst_output_name], {instrument_head.get_inputs()[0].name: inst_input})[0]
    inst_scores = np.asarray(inst_scores).reshape(len(embedding), 40)
    del instrument_head
    inst_labels = instrument_meta['classes']
    inst_summary_scores = inst_scores.mean(axis=0)
    inst_ids = sorted(range(len(inst_labels)), key=lambda i: float(inst_summary_scores[i]), reverse=True)
    instrument_result = {'status': 'model_candidates',
                         'method': 'Discogs-Effnet music embeddings + MTG-Jamendo 40-class instrument head',
                         'summary': [{'label': inst_labels[i], 'mean_score': round(float(inst_summary_scores[i]), 5),
                                      'peak_score': round(float(inst_scores[:, i].max()), 5)}
                                     for i in inst_ids[:10] if inst_labels[i] != 'voice' and inst_summary_scores[i] >= .025],
                         'timeline': [], 'temporal_resolution_seconds': 4.096,
                         'score_meaning': 'Independent multi-label music tag scores, not calibrated probabilities or exact source onsets.'}
    for first in range(0, len(embedding), 2):
        start = frame_starts[first] * .016
        end = min(duration, (frame_starts[min(first + 1, len(frame_starts) - 1)] + 128) * .016)
        if start >= duration:
            continue
        mean = inst_scores[first:first + 2].mean(axis=0)
        ids = sorted(range(len(inst_labels)), key=lambda i: float(mean[i]), reverse=True)
        instrument_result['timeline'].append({'start': round(start, 6), 'end': round(end, 6),
                                              'candidates': [{'label': inst_labels[i], 'score': round(float(mean[i]), 5)}
                                                             for i in ids[:7] if inst_labels[i] != 'voice' and mean[i] >= .05]})
    provenance.append({'name': instrument_meta['name'], 'weights': str(instrument_path),
                       'sha256': digest(instrument_path), 'bytes': instrument_path.stat().st_size,
                       'official_source': instrument_meta['link'],
                       'download_source': 'https://github.com/coiluck/kokone-music-for-mobile/tree/9cfeb133ba8ea25385a239834395c3a28645a93c/src-tauri/models',
                       'license': 'CC BY-NC-SA 4.0', 'device': 'cpu'})
    for mood in MOODS:
        path = folder / ('mood_' + mood + '-discogs-effnet-1.onnx')
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        head = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        shape = head.get_inputs()[0].shape
        inputs = embedding
        if len(shape) == 4:
            inputs = inputs[:, None, None, :]
        elif len(shape) != 2:
            raise ValueError('Unexpected mood model input shape: ' + str(shape))
        values = head.run(None, {head.get_inputs()[0].name: inputs})[0]
        values = np.asarray(values).reshape(len(embedding), -1)
        all_scores[mood] = values[:, meta['classes'].index(mood)]
        provenance.append({'name': meta['name'], 'weights': str(path), 'sha256': digest(path),
                           'bytes': path.stat().st_size, 'official_source': meta['link'],
                           'download_source': 'https://huggingface.co/rlupi/deep-cuts-models-noncommercial',
                           'license': 'CC BY-NC-SA 4.0', 'device': 'cpu'})
        del head
    # Average four approximately 2s patches for a stable editorial mood hint.
    timeline = []
    for first in range(0, len(embedding), 4):
        start = frame_starts[first] * .016
        end = min(duration, (frame_starts[min(first + 3, len(frame_starts) - 1)] + 128) * .016)
        if start >= duration:
            continue
        values = {m: round(float(np.mean(all_scores[m][first:first + 4])), 5) for m in MOODS}
        strongest = sorted(values, key=values.get, reverse=True)
        timeline.append({'start': round(start, 6), 'end': round(end, 6), 'scores': values,
                         'leading_candidates': [m for m in strongest[:2] if values[m] >= .55]})
    averages = {m: round(float(np.mean(all_scores[m])), 5) for m in MOODS}
    return {'status': 'model_candidates', 'method': 'Discogs-Effnet music embeddings + Essentia trained mood heads',
            'summary_scores': averages, 'timeline': timeline,
            'score_meaning': 'Independent binary-classifier scores, not mutually exclusive emotions or calibrated human judgments.',
            'scope': 'Perceived musical mood candidates; not performer intent, lyrical meaning or listener emotion.',
            'frontend': {'implementation': 'NumPy recreation of published Essentia MusiCNN frontend',
                         'sample_rate': 16000, 'frame_size': 512, 'hop_size': 256, 'mel_bands': 96,
                         'patch_frames': 128, 'tail': 'Repeat final mel frame to complete final patch.',
                         'reference_check': {'reference': 'Official essentia.js 0.1.3 WASM TensorflowInputMusiCNN',
                                             'cases': ['silence', '440Hz sine', 'seeded white noise'],
                                             'frame_feature_max_absolute_error': 1.895427703857422e-5,
                                             'scope': '512-sample frame features checked; full neural pipeline equivalence not_verified.'}},
            'speech_domain_warning': 'Music mood heads may be out of domain for spoken-only or very short clips.'}, [
                {'name': 'Discogs-Effnet', 'weights': str(backbone_path), 'sha256': digest(backbone_path),
                 'bytes': backbone_path.stat().st_size,
                 'official_source': 'https://essentia.upf.edu/models/feature-extractors/discogs-effnet/',
                 'download_source': 'https://huggingface.co/rlupi/deep-cuts-models-noncommercial',
                 'license': 'CC BY-NC-SA 4.0', 'device': 'cpu'}, *provenance], instrument_result, music_voice


def stems(receipt_path, source_sha, duration):
    if not receipt_path:
        return {'status': 'not_provided'}
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('sha256') != source_sha:
        raise ValueError('Structure receipt canonical WAV SHA256 mismatch.')
    directory = Path(receipt['demix_directory'])
    output = {}
    for name in ('vocals', 'drums', 'bass', 'other'):
        found = sorted(directory.rglob(name + '.wav'))
        if len(found) != 1:
            continue
        samples, actual = load_wav(found[0])
        if abs(actual - duration) > .1:
            raise ValueError('Separated stem duration mismatch.')
        rows = []
        for start in range(0, len(samples), 16000):
            chunk = samples[start:start + 16000]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            rows.append({'start': start / 16000, 'end': min(duration, (start + len(chunk)) / 16000),
                         'rms_dbfs': None if rms <= 0 else round(20 * math.log10(rms), 3)})
        output[name] = {'source': str(found[0]), 'timeline': rows}
    return {'status': 'measured' if output else 'not_found', 'stems': output,
            'meaning': 'Separated-source RMS energy only; separation leakage is not confirmed source presence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-root', type=Path, default=Path('D:/SelfMedia/管理/工具/bgm-analysis'))
    parser.add_argument('--structure-receipt', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh path.')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    started = time.monotonic()
    samples, duration = load_wav(args.wav)
    source_sha = digest(args.wav)
    general_instruments, vocals, tag_model = sound_tags(samples, duration, args.runtime_root, args.threads)
    emotion, mood_models, instruments, music_voice = moods(samples, duration, args.runtime_root, args.threads)
    instruments['supporting_yamnet_summary'] = general_instruments['summary']
    vocals['music_voice_timeline'] = music_voice
    vocals['method'] = 'YAMNet speech/singing classes + Discogs-Effnet voice/instrumental music classifier'
    vocals['score_meaning'] = ('vocal_score is the maximum of two uncalibrated voice-evidence scores; '
                              'it is a permissive ASR-routing score, not a calibrated vocal-presence probability. '
                              'speech_score/singing_score remain YAMNet classes, music_voice_score does not distinguish speech from singing.')
    for window in vocals['timeline']:
        middle = (window['start'] + window['end']) / 2
        evidence = next((v for v in music_voice if v['start'] <= middle < v['end']), None)
        score = evidence['voice_score'] if evidence else 0
        window['yamnet_vocal_score'] = window['vocal_score']
        window['music_voice_score'] = score
        window['music_voice_context'] = [evidence['start'], evidence['end']] if evidence else None
        window['vocal_score'] = max(window['vocal_score'], score)
        window['model_disagreement'] = window['yamnet_vocal_score'] < .1 and score >= .6
        if window['activity'] in ('uncertain', 'no_vocal_evidence') and score >= .3:
            window['activity'] = 'uncertain_vocal_candidate'
    vocals['fusion'] = {'method': 'Maximum uncalibrated evidence for permissive ASR routing only.',
                        'different_score_scales': True,
                        'interpretation': 'A high music-voice score with low YAMNet evidence is an unresolved model disagreement; do not assert audible speech/singing or lyrics from this alone.',
                        'confirmation': 'Use independent acoustic transcription quality or human listening; separated-stem energy is auxiliary and can include leakage.'}
    vocals['model_disagreement'] = {'flagged_windows': sum(v['model_disagreement'] for v in vocals['timeline']),
                                    'total_windows': len(vocals['timeline']),
                                    'criterion': 'YAMNet explicit vocal max <0.1 while music voice/instrumental score >=0.6.',
                                    'yamnet_mean_score': round(float(np.mean([v['yamnet_vocal_score'] for v in vocals['timeline']])), 5),
                                    'music_voice_mean_score': round(float(np.mean([v['music_voice_score'] for v in vocals['timeline']])), 5)}
    vocals['segments'] = group_activity(vocals['timeline'])
    from speech_emotion import analyze_speech_emotion
    emotion['speech_emotion'] = analyze_speech_emotion(samples, vocals['timeline'], args.runtime_root, args.threads)
    stem_measurements = stems(args.structure_receipt, source_sha, duration)
    music_score = max((v['mean_score'] for v in general_instruments['summary']), default=0)
    speech_mean = float(np.mean([v['speech_score'] for v in vocals['timeline']]))
    singing_mean = float(np.mean([v['singing_score'] for v in vocals['timeline']]))
    if speech_mean > .6 and singing_mean < .2 and music_score < .1:
        emotion['applicability'] = 'low_spoken_audio'
        emotion['leading_candidates'] = []
        for window in emotion['timeline']:
            window['leading_candidates'] = []
        instruments['applicability'] = 'low_spoken_audio'
        instruments['out_of_domain_raw_summary'] = instruments['summary']
        instruments['summary'] = []
        for window in instruments['timeline']:
            window['out_of_domain_raw_candidates'] = window['candidates']
            window['candidates'] = []
    else:
        emotion['applicability'] = 'music_candidate'
        instruments['applicability'] = 'music_candidate'
    result = {'schema': 'bgm-semantics-v1',
              'audio': {'source': str(args.wav.resolve()), 'sha256': source_sha,
                        'duration_seconds': duration, 'time_origin': 'canonical_wav_start'},
              'instruments': instruments, 'vocals': vocals, 'emotion': emotion,
              'stem_measurements': stem_measurements, 'models': [tag_model, *mood_models],
              'implementation': {'script': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
              'elapsed_seconds': round(time.monotonic() - started, 3),
              'limitations': ['Model scores are candidates, not human-listened ground truth.',
                              'Small-window tags can miss buried voices and instruments in a dense mix.',
                              'Mood classifiers do not identify semantic narrative emotion from lyrics.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'duration_seconds': duration,
                      'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
