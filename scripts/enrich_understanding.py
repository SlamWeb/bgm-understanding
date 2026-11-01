"""Join independently measured music, sound-label and transcript reports for agents."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_bound_report(path, audio):
    report = read(path)
    source = report.get('audio', {})
    if source.get('sha256') != audio['sha256']:
        raise ValueError(f'Semantic/transcript report belongs to another WAV: {path}')
    report_duration = float(source.get('duration_seconds', -1))
    if not math.isfinite(report_duration) or abs(report_duration-audio['duration_seconds']) > .02:
        raise ValueError(f'Semantic/transcript report has mismatched coverage: {path}')
    if source.get('time_origin') != 'canonical_wav_start':
        raise ValueError(f'Semantic/transcript time origin differs: {path}')
    if Path(source.get('source', '')).resolve() != Path(audio['file']).resolve():
        raise ValueError(f'Semantic/transcript source path differs: {path}')
    coverage = report.get('coverage')
    if coverage is not None:
        start = float(coverage.get('start', coverage.get('selection_start_seconds', -1)))
        end = float(coverage.get('end', coverage.get('selection_end_seconds', -1)))
        if not math.isfinite(start) or not math.isfinite(end) or abs(start) > 1e-6 or abs(end-report_duration) > .02:
            raise ValueError(f'Enrichment only covers an excerpt of the canonical WAV: {path}')
    return report


def validate_times(value, duration):
    if isinstance(value, list):
        for child in value:
            validate_times(child, duration)
    elif isinstance(value, dict):
        if 'start' in value and 'end' in value:
            start, end = value['start'], value['end']
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [start, end]):
                raise ValueError('Nonfinite interval in enrichment.')
            if not 0 <= start <= end <= duration+.02:
                raise ValueError(f'Enrichment interval outside WAV: {start}, {end}')
        for child in value.values():
            validate_times(child, duration)


def compact_vocals(value, limit=64):
    """Keep exact activity intervals separately; pool dense context scores for agent reading."""
    result = dict(value)
    raw = result.get('timeline', [])
    if len(raw) <= limit:
        return result
    step = math.ceil(len(raw)/limit)
    timeline = []
    scores = ['vocal_score', 'speech_score', 'singing_score', 'yamnet_vocal_score', 'music_voice_score']
    for first in range(0, len(raw), step):
        group = raw[first:first+step]
        weights = [row['end']-row['start'] for row in group]
        total = sum(weights)
        if total <= 0:
            raise ValueError('Invalid vocal context coverage.')
        row = {'start': group[0]['start'], 'end': group[-1]['end']}
        for key in scores:
            if all(isinstance(v.get(key), (int, float)) for v in group):
                row[key] = round(sum(v[key]*w for v,w in zip(group,weights))/total, 5)
        row['vocal_score_peak'] = max(v.get('vocal_score', 0) for v in group)
        row['model_disagreement'] = any(v.get('model_disagreement', False) for v in group)
        row['model_disagreement_fraction'] = round(sum(w for v,w in zip(group,weights) if v.get('model_disagreement', False))/total, 5)
        timeline.append(row)
    result['timeline'] = timeline
    result.pop('music_voice_timeline', None)
    for row in timeline:
        row.pop('yamnet_vocal_score', None)
        row.pop('music_voice_score', None)
    result['summary_window_limit'] = limit
    result['raw_window_count'] = len(raw)
    result['summary_note'] = 'Duration-weighted score summaries; use original activity segments and raw semantics for precise voice entrances.'
    return result


def compact_instruments(value):
    result = dict(value)
    rows = []
    for original in value.get('timeline', []):
        row = dict(original)
        for key in ['candidates', 'instruments', 'top_candidates']:
            if isinstance(row.get(key), list):
                row[key] = row[key][:3]
        rows.append(row)
    result['timeline'] = rows
    result['per_window_candidate_limit'] = 3
    return result


def compact_dynamics(value):
    result = dict(value)
    result['windows'] = []
    for original in value.get('windows', []):
        row = dict(original)
        row['band_rms'] = {key: float(f'{level:.5g}') for key,level in row.get('band_rms', {}).items()}
        result['windows'].append(row)
    result['change_candidates'] = []
    for original in value.get('change_candidates', []):
        row = {key: val for key,val in original.items() if key != 'before_after'}
        if 'frequency_share_delta' in row:
            row['frequency_share_delta'] = {key: float(f'{val:.5g}') for key,val in row['frequency_share_delta'].items()}
        for key in ['score', 'level_delta_db']:
            if isinstance(row.get(key), (float,int)):
                row[key] = float(f'{row[key]:.5g}')
        result['change_candidates'].append(row)
    result['detail_note'] = 'Agent-facing summaries; exact band values and before/after diagnostic windows remain in raw_files.features.'
    return result


def compact_transcript(value):
    result = dict(value)
    result['segments'] = []
    for original in value.get('segments', []):
        row = dict(original)
        words = row.pop('words', None)
        if words is not None:
            row['word_count'] = len(words)
        result['segments'].append(row)
    result['word_timing_note'] = 'Phrase times retained here; full word/token timestamps and diagnostics remain in the referenced raw report.'
    return result


def build(core_path, semantics_path=None, lyrics_path=None, corrections_path=None, speech_path=None, lyrics_skipped=False):
    if lyrics_skipped and (lyrics_path or corrections_path):
        raise ValueError('Skipped transcription cannot also provide lyrics or corrections.')
    core = read(core_path)
    if digest(core['audio']['file']) != core['audio']['sha256']:
        raise ValueError('Core WAV changed before enrichment.')
    if digest(core['audio']['original_file']) != core['audio']['original_sha256']:
        raise ValueError('Original audio changed before enrichment.')
    duration = core['audio']['duration_seconds']
    semantics = load_bound_report(semantics_path, core['audio']) if semantics_path else None
    transcript = load_bound_report(lyrics_path, core['audio']) if lyrics_path else None
    for value in [semantics, transcript]:
        if value:
            validate_times(value, duration)
    result = dict(core)
    result['dynamics'] = compact_dynamics(result['dynamics'])
    result['schema'] = 'bgm-understanding-v2'
    result['core_status'] = core['status']
    missing = {'status': 'not_analyzed', 'reason': 'Stage skipped, failed or exceeded its resource budget; consult stage log.'}
    for name in ['instruments', 'vocals', 'emotion']:
        result[name] = semantics.get(name, dict(missing)) if semantics else dict(missing)
    result['vocals'] = compact_vocals(result['vocals'])
    result['instruments'] = compact_instruments(result['instruments'])
    if result['emotion'].get('applicability') == 'low_spoken_audio':
        result['rhythm'] = {**result['rhythm'], 'applicability': 'low_spoken_audio',
                            'use_note': 'Speech-dominant audio: prioritize phrase timing rather than an estimated musical beat grid.'}
        result['structure'] = {**result['structure'], 'applicability': 'low_spoken_audio'}
    if transcript:
        # Keep recognition evidence/quality; never replace uncertain words with song-title knowledge.
        result['lyrics'] = compact_transcript({k: v for k, v in transcript.items() if k not in ['audio', 'raw_segments', 'rejected_segments', 'fragments']})
        rejected = transcript.get('rejected_segments', [])
        weak_reasons = {'low_decoder_logprob', 'decoder_no_speech_with_weak_text'}
        result['lyrics']['uncertain_segments'] = [
            {**row, 'source': 'decoder_filter_rejection', 'needs_review': True,
             'use_for_exact_subtitles': False}
            for row in rejected
            if row.get('text', '').strip() and row.get('reasons')
            and set(row['reasons']).issubset(weak_reasons)
        ][:12]
        result['lyrics']['rejected_count'] = len(rejected)
        result['lyrics']['fragment_count'] = len(transcript.get('fragments', []))
        result['lyrics']['uncertain_segments_meaning'] = 'Lower-support hypotheses rejected by decoder quality rules; not confirmed lyrics. Repetitive or voice-unsupported output stays only in raw report.'
        result['lyrics']['raw_report_file'] = str(Path(lyrics_path).resolve())
        result['lyrics']['meaning'] = '自动转写的说话或歌词候选；按句/短语提供起止秒数，不保证诗句、歌词或逐字边界正确。空列表表示未可靠检出文字。'
    else:
        result['lyrics'] = {**missing, 'segments': []}
        if lyrics_skipped:
            result['lyrics'] = {'status': 'skipped_not_requested', 'segments': [],
                                'reason': 'Transcription explicitly disabled; no ASR model was run.',
                                'requested': False}
    if corrections_path:
        corrections = load_bound_report(corrections_path, core['audio'])
        validate_times(corrections, duration)
        if not lyrics_path or corrections.get('raw_lyrics_sha256') != digest(lyrics_path):
            raise ValueError('Context corrections describe a different ASR report.')
        for change in corrections.get('segments', []):
            matches = [segment for segment in result['lyrics']['segments']
                       if abs(segment['start']-change['start']) < .005
                       and abs(segment['end']-change['end']) < .005
                       and segment['text'] == change['raw_text']]
            if len(matches) != 1:
                raise ValueError('Context correction does not match one original ASR phrase.')
            matches[0]['normalized_text_candidate'] = change['normalized_text_candidate']
            matches[0]['normalization'] = {'source': 'agent_contextual_correction', 'audio_verified': False,
                                            'basis': corrections.get('basis'), 'needs_review': True,
                                            'timing_note': 'Phrase interval inherited from raw ASR; corrected characters have not been independently aligned.'}
        result['lyrics']['contextual_corrections_file'] = str(Path(corrections_path).resolve())
    if speech_path:
        speech = load_bound_report(speech_path, core['audio'])
        validate_times(speech, duration)
        result['emotion']['speech_classifier'] = speech.get('speech_emotion', dict(missing))
        result['lyrics']['secondary_transcript'] = compact_transcript(speech.get('secondary_transcript', dict(missing)))
        result['lyrics']['secondary_model'] = speech.get('model', speech.get('models', {}))
    statuses = {key: result[key].get('status', 'unknown') for key in ['instruments', 'vocals', 'emotion', 'lyrics']}
    success = {'complete', 'success', 'analyzed', 'ok', 'model_candidates', 'no_detected_words', 'no_vocals_detected', 'no_speech_detected', 'no_credible_text', 'no_reliable_text', 'no_vocals', 'no_text_detected', 'no_supported_transcript'}
    result['analysis_status'] = {'core': core['status'], **statuses}
    result['status'] = 'complete' if core['status'] == 'complete' and all(
        v in success or (key == 'lyrics' and lyrics_skipped and v == 'skipped_not_requested')
        for key, v in statuses.items()) else 'partial'
    result['completion_meaning'] = 'complete 表示所有请求的阶段已运行并产出结果，包括未检出文字；不代表预测经人工确认。'
    result['uncertainty'] = {
        **core['uncertainty'],
        'instruments': 'model_candidates' if semantics else 'not_analyzed',
        'lyrics': 'automatic_transcription_candidates' if transcript else 'not_analyzed',
        'emotion': 'model_mood_candidates' if semantics else 'not_analyzed',
        'human_listened': False,
    }
    result['editing_use'].update({
        'instruments': '乐器窗口是该时间范围内的模型候选；不能把窗口内每个起音都归给该乐器。',
        'vocals': '参考人声活动范围安排人物/字幕；分轨残留、哼唱和人声分类都可能误判。',
        'lyrics': '素材可呼应有质量提示的歌词/说话短语；需逐字字幕或精确唱词时另行复核与对齐。',
        'emotion': '情绪是音乐标签模型的候选分数，结合选题解释；分数非校准概率，能量不等于情绪。',
    })
    result['raw_files'].update({
        'core_handoff': str(Path(core_path).resolve()),
        'semantics': str(Path(semantics_path).resolve()) if semantics_path else None,
        'transcript': str(Path(lyrics_path).resolve()) if lyrics_path else None,
        'speech_semantics': str(Path(speech_path).resolve()) if speech_path else None,
    })
    if semantics:
        result['semantic_models'] = semantics.get('models', semantics.get('model', {}))
        if 'limitations' in semantics:
            result['uncertainty']['semantic_limitations'] = semantics['limitations']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['core', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['semantics', 'lyrics', 'corrections', 'speech']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--lyrics-skipped', action='store_true', help='Transcription intentionally not requested, rather than failed.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new file.')
    result = build(args.core, args.semantics, args.lyrics, args.corrections, args.speech, args.lyrics_skipped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'status': result['status']}))


if __name__ == '__main__':
    main()
