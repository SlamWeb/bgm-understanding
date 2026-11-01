"""Validate an agent-facing BGM file's integrity, coverage, and time contract.

This is a mechanical validator, not evidence that a transcription, instrument,
emotion, beat, or segment prediction is perceptually correct.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(document, expected_audio=None, require_enriched=False, decode_source=False):
    if document.get('schema') in ['bgm-editing-brief-v1','bgm-editing-brief-v2']:
        from summarize_bgm import validate_brief
        return validate_brief(document,expected_audio,require_enriched,decode_source)
    errors, warnings, checks = [], [], []

    def check(ok, message):
        (checks if ok else errors).append(message)

    audio = document.get('audio', {})
    duration = audio.get('duration_seconds')
    check(finite(duration) and duration > 0, 'positive finite audio.duration_seconds')
    if errors:
        return {'valid': False, 'errors': errors, 'warnings': warnings, 'checks': checks}
    eps = .025
    check(document.get('status') in ('complete', 'partial'), 'explicit complete/partial status')
    check(bool(audio.get('time_origin')), 'explicit audio time origin')
    check(finite(audio.get('source_start_seconds')) and audio['source_start_seconds'] >= 0,
          'explicit nonnegative original-file offset')
    original = Path(audio.get('original_file', ''))
    canonical = Path(audio.get('file', ''))
    check(original.is_file(), 'original audio exists')
    check(canonical.is_file(), 'canonical WAV exists')
    if expected_audio is not None:
        check(original.resolve() == Path(expected_audio).resolve(), 'matches requested source path')
    if original.is_file():
        check(sha256(original) == audio.get('original_sha256'), 'original audio hash matches')
    if canonical.is_file():
        check(sha256(canonical) == audio.get('sha256'), 'canonical WAV hash matches')
        try:
            with wave.open(str(canonical), 'rb') as w:
                check(abs(w.getnframes() / w.getframerate() - duration) < .01,
                      'canonical sample duration matches reported duration')
        except (wave.Error, EOFError):
            errors.append('canonical file is not a readable PCM WAV')

    def inspect(node, path=''):
        if isinstance(node, dict):
            for left, right in [('start', 'end'), ('start_seconds', 'end_seconds')]:
                if left in node and right in node and (finite(node[left]) or finite(node[right])):
                    check(finite(node[left]) and finite(node[right]) and
                          -eps <= node[left] < node[right] <= duration + eps,
                          f'{path or "root"}: interval within audio')
            # These are timeline coordinates, not duration or count fields.
            for key in ('seconds', 'change_seconds', 'aligned_bar_start_seconds'):
                if key in node and node[key] is not None:
                    check(finite(node[key]) and -eps <= node[key] <= duration + eps,
                          f'{path}.{key}: point within audio')
            for key, value in node.items():
                inspect(value, f'{path}.{key}' if path else key)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                inspect(value, f'{path}[{i}]')
        elif isinstance(node, float) and not math.isfinite(node):
            errors.append(f'{path}: nonfinite JSON number')

    inspect(document)
    for key in ('rhythm', 'sound_events', 'dynamics', 'structure', 'entry_candidates'):
        check(key in document, f'core field {key} exists')
    for key in ('beats_seconds', 'bar_starts_seconds'):
        points = document.get('rhythm', {}).get(key, [])
        check(isinstance(points, list) and all(finite(x) and 0 <= x < duration for x in points),
              f'rhythm.{key}: finite points within audio')
        if isinstance(points, list) and all(finite(x) for x in points):
            check(points == sorted(set(points)), f'rhythm.{key}: sorted and unique')

    def coverage(rows, name):
        if not rows:
            errors.append(f'{name}: no coverage windows')
            return
        valid_rows = [r for r in rows if isinstance(r, dict) and finite(r.get('start')) and finite(r.get('end'))]
        check(len(valid_rows) == len(rows), f'{name}: numeric start/end windows')
        if len(valid_rows) != len(rows):
            return
        valid_rows.sort(key=lambda r: (r['start'], r['end']))
        check(abs(valid_rows[0]['start']) <= eps, f'{name}: begins at audio start')
        end = valid_rows[0]['end']
        gaps = []
        for row in valid_rows[1:]:
            if row['start'] - end > eps:
                gaps.append([end, row['start']])
            end = max(end, row['end'])
        check(not gaps, f'{name}: no uncovered intervals')
        check(abs(end - duration) <= eps, f'{name}: reaches audio end')

    coverage(document.get('dynamics', {}).get('windows', []), 'dynamics.windows')
    sections = document.get('structure', {}).get('sections', [])
    if sections:
        coverage(sections, 'structure.sections')
    elif document.get('status') == 'complete':
        # A truly silent file may intentionally suppress all structure outputs.
        warnings.append('complete file has no structure sections; confirm silence handling')

    if document.get('uncertainty', {}).get('human_listened') is not False:
        warnings.append('human_listened is not explicitly false; verify any human-validation claim')
    if require_enriched:
        # Accept either direct fields or a named semantic bundle, so this validator
        # can verify the same contract before/after a compact handoff is assembled.
        bundle = document.get('semantics', document.get('audio_semantics', document))
        completed = {'complete', 'success', 'analyzed', 'ok', 'model_candidates'}
        nonfailure_empty = {'no_detected_words', 'no_vocals_detected', 'no_speech_detected',
                            'no_credible_text', 'no_reliable_text', 'no_vocals',
                            'no_text_detected', 'no_supported_transcript', 'no_reliable_lyrics'}
        for key in ('instruments', 'vocals', 'emotion'):
            module = bundle.get(key)
            check(isinstance(module, dict), f'enriched field {key} exists')
            if not isinstance(module, dict):
                continue
            check(bool(module.get('status')), f'{key}: explicit status')
            check(bool(module.get('method') or module.get('model') or module.get('source')),
                  f'{key}: explicit analysis method')
            rows = module.get('timeline', module.get('windows', []))
            if document.get('status') == 'complete':
                check(module.get('status') in completed | nonfailure_empty,
                      f'{key}: complete handoff has a successful analysis status')
            if module.get('status') in completed:
                coverage(rows, f'{key}.timeline')
            if key == 'emotion' and module.get('status') in completed:
                method = json.dumps([module.get(k) for k in ('method', 'model', 'source')], ensure_ascii=False).lower()
                check(not (('rms' in method or 'energy' in method) and
                           not any(s in method for s in ('model', 'essentia', 'mood', 'classifier'))),
                      'emotion has model evidence rather than energy-only relabeling')
        lyrics = document.get('lyrics', document.get('transcript', document.get('transcription')))
        check(isinstance(lyrics, dict), 'transcription/lyrics field exists')
        if isinstance(lyrics, dict):
            check(bool(lyrics.get('status')), 'lyrics: explicit status')
            skipped = lyrics.get('status') == 'skipped_not_requested' and lyrics.get('requested') is False
            if document.get('status') == 'complete':
                check(skipped or lyrics.get('status') in completed | nonfailure_empty,
                      'lyrics: complete handoff has a successful analysis status')
            check(skipped or bool(lyrics.get('method') or lyrics.get('model') or lyrics.get('source')),
                  'lyrics: explicit analysis method')
            segments = lyrics.get('segments', [])
            check(isinstance(segments, list), 'lyrics: segments array')
            if isinstance(segments, list):
                check(all(isinstance(x, dict) and isinstance(x.get('text'), str) and x['text'].strip()
                          for x in segments), 'lyrics: nonempty text for every retained segment')
                if skipped:
                    check(not segments, 'lyrics: skipped stage contains no transcript')
                elif not segments:
                    warnings.append('lyrics has no retained text; inspect whether instrumental or uncertain')
    if decode_source and original.is_file() and canonical.is_file():
        if audio.get('source_start_seconds') != 0:
            errors.append('full-source verification requested for an offset selection')
        else:
            import imageio_ffmpeg
            command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(original),
                       '-map', '0:a:0', '-vn', '-ar', '44100', '-ac', '2', '-f', 's16le', 'pipe:1']
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            digest, count = hashlib.sha256(), 0
            while True:
                chunk = process.stdout.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                count += len(chunk)
            _, stderr = process.communicate()
            check(process.returncode == 0, 'independent original-file decoding succeeds')
            check(abs(count / (44100 * 2 * 2) - duration) < .01, 'whole original audio is covered')
            with wave.open(str(canonical), 'rb') as w:
                if w.getparams()[:3] == (2, 2, 44100):
                    wav_digest = hashlib.sha256()
                    while True:
                        chunk = w.readframes(262144)
                        if not chunk:
                            break
                        wav_digest.update(chunk)
                    check(wav_digest.hexdigest() == digest.hexdigest(),
                          'canonical PCM equals independently decoded full original')
                else:
                    warnings.append('canonical format differs from independent 44.1kHz stereo decoder')
            if stderr:
                warnings.append('independent decoder reported: ' + stderr.decode('utf-8', 'replace')[:500])
    return {'valid': not errors, 'errors': errors, 'warnings': warnings,
            'checks': checks, 'check_count': len(checks),
            'scope': 'Mechanical integrity and coverage only; not human-perception accuracy.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('json', type=Path)
    p.add_argument('--audio', type=Path)
    p.add_argument('--require-enriched', action='store_true')
    p.add_argument('--decode-source', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.output and args.output.exists():
        p.error('Use a new validation receipt path.')
    document = json.loads(args.json.read_text(encoding='utf-8-sig'))
    result = validate(document, args.audio, args.require_enriched, args.decode_source)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)
    raise SystemExit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()
