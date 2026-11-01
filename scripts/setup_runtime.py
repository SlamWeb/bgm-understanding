"""Plan/install a private Python 3.11 CPU runtime; never changes global packages.

No mutation without --apply. Weights and upstream source revisions are hash/commit
locked in runtime.lock.json. Optional legacy transcription is outside this setup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import venv

SKILL = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_target(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes runtime root: ' + relative)
    return path


def matches(path, asset):
    return path.is_file() and path.stat().st_size == asset['bytes'] and digest(path) == asset['sha256']


def download(root, asset):
    path = safe_target(root, asset['path'])
    if path.exists():
        if matches(path, asset):
            return
        raise ValueError('Existing asset differs from lock; preserve and inspect: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.install-part')
    if partial.exists():
        raise FileExistsError('Interrupted download preserved; inspect before retry: ' + str(partial))
    request = urllib.request.Request(asset['url'], headers={'User-Agent': 'bgm-understanding-setup/1'})
    received = 0
    with urllib.request.urlopen(request, timeout=60) as response, partial.open('xb') as output:
        while chunk := response.read(1 << 20):
            received += len(chunk)
            if received > asset['bytes']:
                raise ValueError('Download exceeds locked size: ' + asset['path'])
            output.write(chunk)
    if not matches(partial, asset):
        raise ValueError('Downloaded asset SHA/size mismatch; file retained: ' + str(partial))
    partial.rename(path)


def clone(root, source):
    path = safe_target(root, source['path'])
    if path.exists():
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if head != source['commit']:
            raise ValueError('Existing source revision differs; no checkout performed: ' + str(path))
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
            raise ValueError('Existing upstream source has local changes: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', source['url']], check=True)
    subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', source['commit']], check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', source['commit']], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--profile', choices=['signal', 'full'], default='full')
    parser.add_argument('--apply', action='store_true', help='Create the private venv, fetch sources and verified weights.')
    args = parser.parse_args()
    lock = json.loads((SKILL / 'runtime.lock.json').read_text(encoding='utf-8'))
    root = args.runtime_root.resolve()
    assets = [a for a in lock['assets'] if args.profile == 'full' or a['profile'] == 'signal']
    sources = lock['sources'] if args.profile == 'full' else []
    python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    plan = {'mode': 'apply' if args.apply else 'plan_only', 'profile': args.profile,
            'runtime_root': str(root), 'python': str(python),
            'model_bytes': sum(a['bytes'] for a in assets),
            'download_bytes_missing_or_changed': sum(a['bytes'] for a in assets if not matches(safe_target(root,a['path']),a)),
            'source_revisions': sources,
            'packages': 'Private CPU torch/torchaudio plus requirements-runtime.txt; no transcription models.',
            'note': 'Python packages/cache need additional disk space; model_bytes is not peak RAM or total install size.'}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        return
    if sys.version_info[:2] != (3,11) or platform.machine().lower() not in ('amd64','x86_64'):
        parser.error('Reference installer supports Python 3.11 on Windows/Linux x64.')
    if sys.platform not in ('win32','linux'):
        parser.error('Reference CPU wheel installer supports Windows/Linux only.')
    if sources and not shutil.which('git'):
        parser.error('Git is required to fetch pinned upstream sources.')
    root.mkdir(parents=True, exist_ok=True)
    if not python.is_file():
        venv.create(root / '.venv', with_pip=True)
    env = os.environ.copy()
    env['PIP_CACHE_DIR'] = str(root / 'cache/pip')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    base = [str(python), '-m', 'pip', 'install']
    subprocess.run([*base, '--index-url', 'https://download.pytorch.org/whl/cpu',
                    'torch==' + lock['torch_cpu'], 'torchaudio==' + lock['torchaudio_cpu']], check=True, env=env)
    subprocess.run([*base, '-r', str(SKILL / 'requirements-runtime.txt')], check=True, env=env)
    for source in sources:
        clone(root, source)
    for asset in assets:
        download(root, asset)
    # Required by the legacy runtime layout; Python packages live in the venv.
    for name in ['packages','semantic-packages','temp','all-in-one-trial/packages']:
        safe_target(root, name).mkdir(parents=True, exist_ok=True)
    receipt = {'schema': 'bgm-runtime-install-v1', 'profile': args.profile,
               'python': str(python), 'lock_sha256': digest(SKILL / 'runtime.lock.json'),
               'model_files_verified': len(assets), 'source_commits': sources,
               'validation': 'Downloaded bytes and source identities verified; run doctor and a real audio smoke test.'}
    (root / 'install-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    subprocess.run([str(python), str(SKILL / 'scripts/doctor_runtime.py'), '--runtime-root', str(root),
                    '--profile', args.profile, '--check-imports'], check=True, env=env)
    print('Run understand_bgm.py with this private Python and --tools-dir ' + str(root))


if __name__ == '__main__':
    main()
