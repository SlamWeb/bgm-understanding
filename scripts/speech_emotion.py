"""The audEERING Wav2Small 17K speech-attribute CNN, using existing CPU PyTorch.

Architecture transcribed from the publisher's model card; all frontend kernels
and model tensors are loaded from the publisher's safetensors file. No remote
Python code, Transformers, librosa, or executable checkpoint is loaded.
"""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def _state_dict(path, torch):
    blob = path.read_bytes()
    size = struct.unpack('<Q', blob[:8])[0]
    if not 0 < size < len(blob) - 8:
        raise ValueError('Invalid safetensors header.')
    header = json.loads(blob[8:8 + size])
    data = memoryview(blob)[8 + size:]
    types = {'F32': '<f4', 'I64': '<i8'}
    result = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        start, end = info['data_offsets']
        if not 0 <= start <= end <= len(data) or info['dtype'] not in types:
            raise ValueError('Invalid tensor bounds or dtype.')
        value = np.frombuffer(data[start:end], dtype=types[info['dtype']]).reshape(info['shape']).copy()
        result[name] = torch.from_numpy(value)
    return result


def load_model(path, threads=1):
    import torch
    from torch import nn
    torch.set_num_threads(threads)

    class Conv(nn.Module):
        def __init__(self, first, last):
            super().__init__()
            self.conv = nn.Conv2d(first, last, 3, stride=1, padding=1, bias=False)
            self.norm = nn.BatchNorm2d(last)

        def forward(self, x):
            return torch.relu_(self.norm(self.conv(x)))

    class Spectrogram(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_real = nn.Conv1d(1, 33, 64, stride=32, bias=False)
            self.conv_imag = nn.Conv1d(1, 33, 64, stride=32, bias=False)

        def forward(self, x):
            x = x[:, None, :]
            return self.conv_real(x) ** 2 + self.conv_imag(x) ** 2

    class LogmelFilterBank(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('melW', torch.zeros(33, 26))
            self.register_buffer('amin', torch.tensor([1e-10]))

        def forward(self, x):
            x = torch.matmul(x[:, None, :, :].transpose(2, 3), self.melW)
            return 10 * torch.log10(torch.where(x > self.amin, x, self.amin))

    class Vgg7(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Conv(1, 13)
            for index in range(2, 8):
                setattr(self, 'l' + str(index), Conv(13, 13))
            self.maxpool_A = nn.MaxPool2d(3, stride=2, padding=1)
            self.lin = nn.Conv2d(13, 13, 1)
            self.sof = nn.Conv2d(13, 13, 1)
            self.spectrogram_extractor = Spectrogram()
            self.logmel_extractor = LogmelFilterBank()

        def forward(self, x):
            x = x - x.mean(1, keepdim=True)
            x = x / torch.sqrt((x * x).mean(1, keepdim=True) + 1e-7)
            x = self.logmel_extractor(self.spectrogram_extractor(x))
            x = self.l3(self.l2(self.l1(x)))
            x = self.maxpool_A(x)
            for index in range(4, 8):
                x = getattr(self, 'l' + str(index))(x)
            x = (self.lin(x) * self.sof(x).softmax(2)).sum(2)
            return torch.cat([x, torch.bmm(x, x.transpose(1, 2))], 2).reshape(-1, 338)

    class Wav2Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.vgg7 = Vgg7()
            self.adv = nn.Linear(338, 3)

        def forward(self, x):
            return self.adv(self.vgg7(x))

    model = Wav2Small()
    model.load_state_dict(_state_dict(path, torch), strict=True)
    return model.eval()


def analyze_speech_emotion(samples, vocal_windows, runtime_root, threads=1):
    import torch
    path = Path(runtime_root) / 'models/wav2small/model.safetensors'
    spoken = [v for v in vocal_windows if v['speech_score'] >= .3]
    method = 'audEERING Wav2Small2.0 17K-parameter CNN; arousal/dominance/valence regression'
    base = {'status': 'not_applicable', 'method': method, 'timeline': [],
            'scope': 'Acoustic speech attributes; not lyrics sentiment, music mood, or a speaker psychological state.',
            'dimensions': {'arousal': 'Vocal activation/intensity estimate.',
                           'dominance': 'Vocal dominance/assertiveness estimate.',
                           'valence': 'Positive versus negative expressed vocal affect estimate.'},
            'score_meaning': 'Continuous regression outputs with publisher expected range [0,1]; not class probabilities.',
            'language_validation': 'Release card does not establish Mandarin validation; paper evaluates MSP-Podcast speech. Treat Mandarin transfer as unvalidated.',
            'model': {'name': 'audeering/wav2small', 'source': 'https://huggingface.co/audeering/wav2small',
                      'weights': str(path), 'weights_bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'license': 'CC BY-NC-SA 4.0',
                      'architecture_source': 'Publisher model card; strict state-dict loading verifies architecture/tensor correspondence.',
                      'device': 'cpu'}}
    if not spoken:
        base['reason'] = 'No explicit speech-class evidence; singing-only/unknown vocal clips are outside this speech analysis route.'
        return base
    model = load_model(path, threads)
    duration = len(samples) / 16000
    rows = []
    with torch.inference_mode():
        for sample_start in range(0, len(samples), 64000):
            sample_end = min(sample_start + 64000, len(samples))
            start, end = sample_start / 16000, sample_end / 16000
            overlap = sum(max(0, min(end, v['end']) - max(start, v['start'])) for v in spoken)
            if end - start < .5 or overlap < .5 * (end - start):
                continue
            signal = np.asarray(samples[sample_start:sample_end], dtype=np.float32).copy()
            values = model(torch.from_numpy(signal[None])).numpy()[0]
            if not np.isfinite(values).all():
                raise ValueError('Non-finite speech emotion estimate.')
            rows.append({'start': round(start, 6), 'end': round(min(duration, end), 6),
                         'arousal': round(float(values[0]), 5), 'dominance': round(float(values[1]), 5),
                         'valence': round(float(values[2]), 5),
                         'speech_coverage_fraction': round(overlap / (end - start), 4)})
    if rows:
        base.update({'status': 'model_candidates', 'timeline': rows,
                     'summary': {key: round(float(np.mean([r[key] for r in rows])), 5)
                                 for key in ('arousal', 'dominance', 'valence')}})
    else:
        base['reason'] = 'Speech evidence did not cover enough of a >=0.5s window.'
    return base
