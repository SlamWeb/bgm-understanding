"""Build the compact four-part agent handoff from same-audio measured reports."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def finite_times(values, duration):
    values = [float(x) for x in values]
    if any(not math.isfinite(x) or x < 0 or x > duration + .1 for x in values):
        raise ValueError('Invalid event times.')
    return sorted(set(round(x, 6) for x in values if x < duration))


def grid_tempo(beats):
    """Multi-beat intervals reduce 20ms frame quantization in individual beats."""
    if len(beats) < 2:
        return None
    lag = min(4, len(beats)-1)
    period = statistics.median((beats[i+lag]-beats[i])/lag for i in range(len(beats)-lag))
    return round(60/period, 2) if period > 0 else None


def select_grid(signal, structure, duration):
    dedicated = finite_times(signal['beatSeconds'], duration)
    dedicated_bars = finite_times(signal['downbeatSeconds'], duration)
    if not structure:
        return dedicated, dedicated_bars, 'Beat This', 'dedicated_tracker_only'
    model = finite_times(structure['beats'], duration)
    model_bars = finite_times(structure['downbeats'], duration)
    if len(model)<2:
        return dedicated, dedicated_bars, 'Beat This', 'structure_grid_unresolved'
    if len(dedicated)>=2:
        one,two=grid_tempo(dedicated),grid_tempo(model)
        agreement=abs(one-two)/two<=.08
        intervals=[b-a for a,b in zip(dedicated,dedicated[1:])]
        middle=statistics.median(intervals)
        stable=statistics.median(abs(v-middle) for v in intervals)/middle<=.15
        missing_edges=max(model[0]-dedicated[0],dedicated[-1]-model[-1])
        if agreement and stable and missing_edges>=max(1,2*60/two):
            return dedicated,dedicated_bars,'Beat This','tempo_agrees_and_preserves_missing_edges'
    return model,model_bars,'All-In-One','structure_grid_retained; independent track available in detail'


def compact_windows(raw, limit=48):
    """Pool channel-power measurements with duration weights; preserve tail coverage."""
    step=max(1, math.ceil(len(raw)/limit))
    result=[]
    for first in range(0,len(raw),step):
        group=raw[first:first+step]
        weights=[w['end']-w['start'] for w in group]
        if any(v<=0 for v in weights):
            raise ValueError('Invalid feature interval.')
        total=sum(weights)
        rms=math.sqrt(sum(w['rms']**2*t for w,t in zip(group,weights))/total)
        bands={k:math.sqrt(sum(w['band_rms'][k]**2*t for w,t in zip(group,weights))/total)
               for k in ['low','mid','high']}
        result.append({'start':group[0]['start'],'end':group[-1]['end'],'rms':rms,
                       'dbfs':round(20*math.log10(rms),3) if rms else None,'band_rms':bands})
    return result


def build(analysis_path, receipt_path, features_path, structure_dir=None, structure_note=None):
    a, receipt, features = map(read, [analysis_path, receipt_path, features_path])
    wav = Path(receipt['canonicalWav'])
    sha = digest(wav)
    duration = float(a['audio']['durationSeconds'])
    if a['audio']['sha256'] != sha or features['audio']['sha256'] != sha:
        raise ValueError('Reports do not describe the same WAV bytes.')
    if abs(features['audio']['duration_seconds'] - duration) > .01:
        raise ValueError('Report duration mismatch.')
    if Path(features['audio']['source']).resolve() != wav.resolve():
        raise ValueError('Feature source path mismatch.')
    input_file = Path(receipt['inputFile'])
    if digest(input_file) != receipt['inputSha256']:
        raise ValueError('Original input has changed since decoding.')
    raw_events_path = Path(a['sources']['onsets']['path'])
    if digest(raw_events_path) != a['sources']['onsets']['sha256']:
        raise ValueError('Onset report changed after signal merging.')
    raw_events = read(raw_events_path)
    if Path(raw_events['sourceAudio']).resolve() != wav.resolve():
        raise ValueError('Onset report time/source mismatch.')
    if abs(float(raw_events.get('analysisOffset', 0))) > 1e-8:
        raise ValueError('Expected full canonical WAV onset time origin.')
    accepted = [e for e in raw_events['events'] if e.get('accepted') and 0 <= e['timeSeconds'] < duration]
    silent = bool(a['dynamics']['waveformLevel']['nearSilent'])
    s, sr = None, None
    if structure_dir:
        s, sr = read(Path(structure_dir)/'structure.json'), read(Path(structure_dir)/'receipt.json')
        if sr['sha256'] != sha or abs(sr['duration_seconds']-duration) > .01:
            raise ValueError('Structure does not describe the same WAV.')
        if digest(Path(structure_dir)/'structure.json') != sr['raw_sha256']:
            raise ValueError('Structure output has changed after analysis.')
    # Select a coherent grid, never interleave different beat phases or meters.
    beats,bars,grid_source,grid_reason = select_grid(a['rhythm'],s,duration)
    grid_bpm=grid_tempo(beats)
    sections, boundaries, removed = [], [], []
    if s:
        for seg in s['segments']:
            start, end = float(seg['start']), float(seg['end'])
            if not all(math.isfinite(t) for t in [start, end]) or start < 0 or end < start or end > duration+.1:
                raise ValueError('Invalid structure section range.')
            # Short start/end bookkeeping frames are retained in raw results, not used as semantic boundaries.
            if seg['label'] in ['start','end'] or end-start < .3:
                removed.append(seg)
                continue
            sections.append({'start': round(start, 4), 'end': round(min(end, duration), 4), 'label_candidate': seg['label']})
        for left, right in zip(sections, sections[1:]):
            t = right['start']
            if .3 < t < duration-.3:
                boundaries.append({'seconds': t, 'source': 'allin1', 'labels': [left['label_candidate'],right['label_candidate']]})
        # Cover model bookkeeping gaps at the two edges without moving any internal boundary.
        if sections:
            sections[0]['start']=0.0
            sections[-1]['end']=duration
        else:
            # Some speech/out-of-domain inputs receive only start/end labels.
            # Preserve explicit coverage without inventing a musical section.
            sections = [{'start': 0.0, 'end': duration, 'label_candidate': None,
                         'reason': 'Model did not resolve a meaningful musical section.'}]
    else:
        sections = [{'start':v['rangeSeconds'][0], 'end':v['rangeSeconds'][1], 'label_candidate':None}
                    for v in a['structure']['longSectionCandidates']]
        boundaries = [{'seconds':sec['start'], 'source':'sonara_signal_section'} for sec in sections[1:]]
    candidate_evidence = list(boundaries)
    candidate_evidence += [{'seconds':v['timeSeconds'], 'source':'energy_onset_change', 'types':v['signals']}
                           for v in a['structure']['shortWindowChangeCandidates']]
    candidate_evidence += [{'seconds':v['seconds'], 'source':'waveform_frequency_change', 'types':v['types']}
                           for v in features['changes']]
    for e in candidate_evidence:
        if not math.isfinite(e['seconds']) or not 0 <= e['seconds'] < duration:
            raise ValueError('Invalid change evidence time.')
    # Group nearby measurements without inventing sound events; prioritize structural boundaries.
    seeds = sorted(candidate_evidence, key=lambda e:(0 if e['source']=='allin1' else 1, e['seconds']))
    entries=[]
    for seed in seeds:
        t=seed['seconds']
        if t < .3 or t > duration-.3 or any(abs(t-e['change_seconds']) < 1 for e in entries):
            continue
        support=[v for v in candidate_evidence if abs(v['seconds']-t)<=.75]
        nearby=[v for v in bars if t-.08 <= v <= t+1.2]
        snap=min(nearby,key=lambda v:abs(v-t)) if nearby else None
        entries.append({'change_seconds':round(t,4),'aligned_bar_start_seconds':snap,
                        'evidence':support,'evidence_sources':len(set(v['source'] for v in support))})
    entries=sorted(entries,key=lambda e:(-e['evidence_sources'], 0 if any(x['source']=='allin1' for x in e['evidence']) else 1, e['change_seconds']))[:5]
    windows=[]
    feature_windows=compact_windows(features['windows'])
    max_rms=max((w['rms'] for w in feature_windows),default=0)
    suspicious=[]
    for w in feature_windows:
        if not 0<=w['start']<w['end']<=duration+.01:
            raise ValueError('Invalid feature window coverage.')
        events=sum(w['start']<=e['timeSeconds']<w['end'] for e in accepted)
        row={'start':w['start'],'end':w['end'],'relative_energy':round(w['rms']/max_rms,4) if max_rms else 0,
             'rms_dbfs':w['dbfs'],'band_rms':w['band_rms'],
             'detected_onsets_per_second':round(events/(w['end']-w['start']),3)}
        windows.append(row)
    for w in a['dynamics']['summaryWindows']:
        if w['acceptedOnsetCount']==0 and not silent:
            suspicious.append(w['rangeSeconds'])
    top=sorted(sorted(accepted,key=lambda e:e['strength'],reverse=True)[:32],key=lambda e:e['timeSeconds'])
    if silent:
        beats,bars,entries,boundaries,sections,top=[],[],[],[],[],[]
    result={
        'schema':'bgm-understanding-v1',
        'status':'complete' if s is not None else 'partial',
        'audio':{'file':str(wav),'original_file':str(input_file),'sha256':sha,'duration_seconds':duration,
                 'original_sha256':receipt['inputSha256'],
                 'source_start_seconds':receipt['sourceStartSeconds'],
                 'time_origin':'decoded analysis WAV start; original-file seconds = seconds + source_start_seconds'},
        'rhythm':{'bpm_candidate':None if silent else grid_bpm,
                  'grid_source':grid_source,
                  'grid_selection_reason':grid_reason,
                  'beats_seconds':beats,'bar_starts_seconds':bars,
                  'other_bpm_candidates':[] if silent else [a['rhythm']['tempo']['sonaraBpmCandidate'],a['rhythm']['tempo']['beatThisMedianIntervalBpm']],
                  'tempo_relation':a['rhythm']['tempo']['relation'],
                  'meaning':'模型节拍与小节起拍候选；不等于每一下乐器攻击。'},
        'sound_events':{'events':[{'seconds':e['timeSeconds'],'relative_strength':round(e['strength'],4),'type':'onset'} for e in top],
                        'accepted_total':0 if silent else len(accepted),'all_events_file':str(raw_events_path),
                        'meaning':'本曲内相对起音强度，仅保留较突出候选；未判断乐器或鼓。'},
        'dynamics':{'windows':windows,'raw_window_seconds':.5,'summary_window_limit':48,
                    'change_candidates':[] if silent else features['changes'],
                    'possible_onset_detection_gaps':suspicious,
                    'meaning':'跨声道RMS、归一化能量、低中高频能量及检出密度；密度增加不等于BPM加快。空起音窗口不等于没有声音。'},
        'structure':{'method':'All-In-One via third-party CPU port' if s else 'signal-only fallback',
                     'sections':sections,'boundaries':boundaries,
                     'edge_handling':'Removed short start/end bookkeeping frames; first/last meaningful section extends to actual audio edges, without changing internal boundaries.',
                     'meaning':'变化边界与段落名称均为候选；verse/chorus不直接等于铺垫/高潮。'},
        'entry_candidates':entries,
        'editing_use':{'entry':'entry_candidates按证据支持排序，可用于选择快切进入或重点变化的位置；没有候选也可以从开头编排。',
                       'cuts':'优先参考bar_starts_seconds；需要更密集时参考beats_seconds或声音起点，素材信息多时跨多个小节。',
                       'transitions':'按选题、素材和相邻切点间隔决定转场，避免机械地每拍切换。',
                       'story':'素材顺序、叙事推进与画面可读性由下游agent结合选题决定。'},
        'uncertainty':{'human_listened':False,'instruments':'unconfirmed','lyrics':'not_analyzed','emotion':'not_inferred',
                       'structure_note':structure_note or '已运行模型，但未证明其段落名称与边界符合人类理解。'},
        'raw_files':{'source_receipt':str(Path(receipt_path).resolve()),
                     'signal_analysis':str(Path(analysis_path).resolve()),'features':str(Path(features_path).resolve()),
                     'structure':str((Path(structure_dir)/'structure.json').resolve()) if structure_dir else None},
    }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['analysis','receipt','features','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--structure-dir',type=Path)
    p.add_argument('--structure-note')
    a=p.parse_args()
    if a.output.exists():
        p.error('Output exists; use a new file.')
    result=build(a.analysis,a.receipt,a.features,a.structure_dir,a.structure_note)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(a.output)


if __name__=='__main__':
    main()
