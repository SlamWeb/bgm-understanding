"""MP3/audio to one agent-facing BGM timeline. CPU, local installed runtimes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from runtime_paths import default_runtime_root

HERE = Path(__file__).resolve().parent


def run_stage(name, command, folder, env, max_rss_mb, timeout_seconds):
    import psutil
    log = folder / (name + '.log')
    peak, stopped, reason = 0, False, None
    start = time.monotonic()
    with log.open('x', encoding='utf-8') as output:
        process = subprocess.Popen([sys.executable, *map(str, command)], stdout=output,
                                   stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    children = [parent, *parent.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in children if p.is_running())
                    peak = max(peak, rss)
                    if rss > max_rss_mb * 1_000_000 or time.monotonic() - start > timeout_seconds:
                        reason = 'memory_budget' if rss > max_rss_mb * 1_000_000 else 'timeout'
                        stopped = True
                        for child in reversed(children):
                            try:
                                child.kill()
                            except psutil.Error:
                                pass
                except psutil.Error:
                    pass
                time.sleep(.1)
            code = process.wait()
        except BaseException:
            try:
                parent = psutil.Process(process.pid)
                for child in reversed([parent, *parent.children(recursive=True)]):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            process.wait()
            raise
    receipt = {'stage': name, 'exit_code': code, 'elapsed_seconds': round(time.monotonic()-start, 2),
               'peak_process_tree_rss_bytes': peak, 'stopped_by_budget': stopped,
               'stop_reason': reason, 'log': str(log)}
    (folder / (name + '-receipt.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--tools-dir', type=Path, default=default_runtime_root())
    p.add_argument('--structure-runtime', type=Path)
    p.add_argument('--structure-mode', choices=['full', 'off'], default='full')
    p.add_argument('--structure-model', choices=['harmonix-all', 'harmonix-fold0'], default='harmonix-all')
    p.add_argument('--semantics-mode', choices=['full', 'off'], default='full')
    p.add_argument('--lyrics-mode', choices=['full', 'off'], default='off', help='Optional transcription; disabled by default.')
    p.add_argument('--asr-model', choices=['tiny', 'base', 'small'], default='base')
    p.add_argument('--asr-language', help='Optional language code supported by Whisper; use audio evidence rather than the filename.')
    p.add_argument('--speech-mode', choices=['auto', 'off'], default='off', help='Optional speech ASR/emotion; disabled by default.')
    p.add_argument('--reuse-core', type=Path, help='Reuse verified raw/features/structure from a previous run, writing enrichment to a fresh output directory.')
    p.add_argument('--reuse-structure-run', type=Path, help='Recover verified complete stems after an interrupted structure stage.')
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--duration', type=float)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-rss-mb', type=int, default=2200)
    p.add_argument('--stage-timeout', type=int, default=1800)
    a = p.parse_args()
    if not a.audio.is_file():
        p.error('Input must be a local audio file.')
    if a.output_dir.exists():
        p.error('Use a new output directory; previous analyses are preserved.')
    if not math.isfinite(a.start) or a.start < 0 or (a.duration is not None and (not math.isfinite(a.duration) or not 0 < a.duration <= 600)):
        p.error('start must be nonnegative; duration must be in (0,600].')
    if not 1 <= a.threads <= 8 or a.max_rss_mb < 128 or a.stage_timeout < 1:
        p.error('Invalid resource budget.')
    import psutil  # Fail before creating output if resource supervision is unavailable.
    runtime = a.structure_runtime or a.tools_dir/'all-in-one-trial'
    out = a.output_dir.resolve()
    out.mkdir(parents=True)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        env[key] = str(a.threads)
    core = a.reuse_core.resolve() if a.reuse_core else out
    if a.reuse_core:
        source = json.loads((core/'raw/source-receipt.json').read_text(encoding='utf-8-sig'))
        with a.audio.open('rb') as stream:
            source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if source['inputSha256'] != source_hash or abs(source['sourceStartSeconds']-a.start) > 1e-6:
            p.error('Reused core describes a different input or start offset.')
        if a.duration is not None and abs(source['sourceEndSeconds']-a.start-a.duration) > .05:
            p.error('Reused core describes a different selected duration.')
        if a.duration is None and source.get('requestedSelection') != 'whole_file':
            p.error('Reused core is not attested as a whole-file request; specify its exact --duration or make a fresh full analysis.')
    else:
        signal_cmd = [HERE/'analyze_bgm.py', '--audio', a.audio.resolve(), '--output-dir', out/'raw', '--start', str(a.start), '--tools-dir', a.tools_dir]
        if a.duration is not None:
            signal_cmd += ['--duration', str(a.duration)]
        receipt = run_stage('signals', signal_cmd, out, env, a.max_rss_mb, a.stage_timeout)
        if receipt['exit_code'] or receipt['stopped_by_budget']:
            raise RuntimeError('Signal analysis failed; inspect signals.log. No completed output was created.')
        features = run_stage('features', [HERE/'analyze_signal_features.py', '--wav', out/'raw/canonical.wav', '--output', out/'signal-features.json'], out, env, a.max_rss_mb, a.stage_timeout)
        if features['exit_code'] or features['stopped_by_budget']:
            raise RuntimeError('Feature analysis failed; inspect features.log. No completed output was created.')
    structure = None
    structure_ok = False
    if a.structure_mode == 'full':
        if a.reuse_core and (core/'structure/receipt.json').is_file():
            structure_ok = True  # build_understanding verifies WAV and output hashes below.
        else:
            structure_cmd = [HERE/'analyze_structure.py', '--wav', core/'raw/canonical.wav', '--output-dir', out/'structure', '--runtime-root', runtime, '--model', a.structure_model, '--threads', str(a.threads)]
            if a.reuse_structure_run:
                structure_cmd += ['--reuse-run', a.reuse_structure_run]
            structure = run_stage('structure', structure_cmd, out, env, a.max_rss_mb, a.stage_timeout)
            structure_ok = structure['exit_code'] == 0 and not structure['stopped_by_budget']
    structure_dir = core/'structure' if a.reuse_core and (core/'structure/receipt.json').is_file() else out/'structure'
    build = [HERE/'build_understanding.py', '--analysis', core/'raw/music-analysis.json', '--receipt', core/'raw/source-receipt.json', '--features', core/'signal-features.json', '--output', out/'core-understanding.json']
    if structure_ok:
        build += ['--structure-dir', structure_dir]
    else:
        build += ['--structure-note', 'Explicitly skipped by --structure-mode off.' if structure is None else 'Structure stage failed or exceeded resource budget; inspect structure.log and structure-receipt.json.']
    subprocess.run([sys.executable, *map(str, build)], check=True, env=env)
    enriched = [HERE/'enrich_understanding.py', '--core', out/'core-understanding.json', '--output', out/'bgm-analysis-detail.json']
    if a.lyrics_mode == 'off':
        enriched += ['--lyrics-skipped']
    semantics_ok = False
    if a.semantics_mode == 'full':
        semantic_cmd = [HERE/'analyze_semantics.py', '--wav', core/'raw/canonical.wav', '--output', out/'semantics.json', '--runtime-root', a.tools_dir, '--threads', str(a.threads)]
        if structure_ok:
            semantic_cmd += ['--structure-receipt', structure_dir/'receipt.json']
        semantics = run_stage('semantics', semantic_cmd, out, env, a.max_rss_mb, a.stage_timeout)
        if semantics['exit_code'] == 0 and not semantics['stopped_by_budget']:
            semantics_ok = True
            enriched += ['--semantics', out/'semantics.json']
    if a.lyrics_mode == 'full':
        lyric_cmd = [HERE/'analyze_lyrics.py', '--wav', core/'raw/canonical.wav', '--output', out/'lyrics.json', '--runtime-root', a.tools_dir, '--threads', str(a.threads), '--model', a.asr_model]
        if a.asr_language:
            lyric_cmd += ['--language', a.asr_language]
        if semantics_ok:
            lyric_cmd += ['--semantic-file', out/'semantics.json']
        if structure_ok:
            info = json.loads((structure_dir/'receipt.json').read_text(encoding='utf-8-sig'))
            stems = list(Path(info['demix_directory']).rglob('vocals.wav'))
            if len(stems) == 1:
                lyric_cmd += ['--vocal-wav', stems[0]]
        lyrics = run_stage('lyrics', lyric_cmd, out, env, a.max_rss_mb, a.stage_timeout)
        if lyrics['exit_code'] == 0 and not lyrics['stopped_by_budget']:
            enriched += ['--lyrics', out/'lyrics.json']
    if a.speech_mode == 'auto' and semantics_ok:
        semantic_data = json.loads((out/'semantics.json').read_text(encoding='utf-8-sig'))
        cells = semantic_data.get('vocals', {}).get('timeline', [])
        total = sum(row['end']-row['start'] for row in cells)
        speech_score = sum(row.get('speech_score', 0)*(row['end']-row['start']) for row in cells)/total if total else 0
        singing_score = sum(row.get('singing_score', 0)*(row['end']-row['start']) for row in cells)/total if total else 0
        if speech_score > .5 and speech_score > 2*singing_score:
            speech = run_stage('speech', [HERE/'analyze_speech_semantics.py', '--wav', core/'raw/canonical.wav', '--output', out/'speech-semantics.json', '--runtime-root', a.tools_dir, '--threads', str(a.threads)], out, env, a.max_rss_mb, a.stage_timeout)
            if speech['exit_code'] == 0 and not speech['stopped_by_budget']:
                enriched += ['--speech', out/'speech-semantics.json']
    subprocess.run([sys.executable, *map(str, enriched)], check=True, env=env)
    subprocess.run([sys.executable, str(HERE/'summarize_bgm.py'), '--analysis', str(out/'bgm-analysis-detail.json'), '--output', str(out/'bgm-understanding.json')], check=True, env=env)
    result = json.loads((out/'bgm-understanding.json').read_text(encoding='utf-8'))
    from summarize_bgm import validate_brief, check_editorial_contract
    validation = validate_brief(result, expected_audio=a.audio,
                                require_enriched=semantics_ok,
                                decode_source=a.start == 0 and a.duration is None)
    (out/'final-validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    quality = check_editorial_contract(result)
    quality.update({'bytes': (out/'bgm-understanding.json').stat().st_size,
                    'lines': len((out/'bgm-understanding.json').read_text(encoding='utf-8').splitlines()),
                    'validation': 'Mechanical coverage and output budget; not perceptual accuracy.'})
    (out/'output-quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    if not validation['valid']:
        raise RuntimeError('Final handoff validation failed; inspect final-validation.json.')
    print(json.dumps({'output': str(out/'bgm-understanding.json'), 'status': result['status'],
                      'bytes': quality['bytes'], 'lines': quality['lines'], 'validated': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
