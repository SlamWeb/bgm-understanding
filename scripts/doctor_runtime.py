"""Check pinned assets/source layout without loading models or downloading files."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

SKILL = Path(__file__).resolve().parents[1]


def inspect(root, profile='full', check_imports=False):
    lock = json.loads((SKILL / 'runtime.lock.json').read_text(encoding='utf-8'))
    checks = []
    def check(name, ok, detail=None):
        checks.append({'name': name, 'ok': bool(ok), **({'detail': detail} if detail else {})})
    check('python_3_11', sys.version_info[:2] == (3,11), sys.version.split()[0])
    for asset in lock['assets']:
        if profile == 'signal' and asset['profile'] != 'signal':
            continue
        path = root / asset['path']
        ok = path.is_file() and path.stat().st_size == asset['bytes']
        if ok:
            with path.open('rb') as stream:
                ok = hashlib.file_digest(stream, 'sha256').hexdigest() == asset['sha256']
        check(asset['path'], ok)
    if profile == 'full':
        for source in lock['sources']:
            try:
                result = subprocess.run(['git','-C',str(root/source['path']),'rev-parse','HEAD'],
                                        capture_output=True,text=True,timeout=10)
                check(source['path'], result.returncode == 0 and result.stdout.strip() == source['commit'])
            except (OSError, subprocess.TimeoutExpired) as exc:
                check(source['path'], False, str(exc))
    search_paths = [root/'packages',root/'semantic-packages',root/'all-in-one-trial/packages',
                    root/'all-in-one-trial/infer-source/src',root/'all-in-one-trial/madmom-source',
                    root/'all-in-one-trial/demucs-source',root/'panns-test/yamnet-source']
    sys.path[:0] = [str(p) for p in search_paths]
    modules = ['numpy','scipy','torch','torchaudio','sonara','beat_this','imageio_ffmpeg','psutil']
    if profile == 'full':
        modules += ['onnxruntime','librosa','hydra','omegaconf','julius','soundfile','allin1_infer',
                    'madmom_infer','demucs_infer','torch_audioset']
    for name in modules:
        try:
            check('module:' + name, importlib.util.find_spec(name) is not None)
        except (ImportError,ValueError) as exc:
            check('module:' + name, False, str(exc))
    if check_imports:
        # Child process catches native-DLL/version failures. It imports code only;
        # no checkpoints are loaded and no audio inference is performed.
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(map(str,search_paths)) + os.pathsep + env.get('PYTHONPATH','')
        code = ('import numpy,scipy,torch,torchaudio,sonara,beat_this,imageio_ffmpeg,psutil;'
                'from beat_this.inference import Audio2Beats;'
                + ('import onnxruntime,librosa,hydra,omegaconf,julius,soundfile;' if profile == 'full' else '')
                + 'print("imports_ok")')
        result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env,timeout=90)
        check('native_import_smoke', result.returncode == 0, result.stderr[-1200:] if result.returncode else None)
    return {'schema':'bgm-runtime-doctor-v1','runtime_root':str(root),'profile':profile,
            'ready':all(c['ok'] for c in checks),'checks':checks,
            'scope':'Asset hashes, source revisions and module discovery/imports; not full inference or perceptual validation.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--profile',choices=['signal','full'],default='full')
    parser.add_argument('--check-imports',action='store_true')
    args=parser.parse_args()
    report=inspect(args.runtime_root.resolve(),args.profile,args.check_imports)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    sys.exit(0 if report['ready'] else 1)


if __name__ == '__main__':
    main()
