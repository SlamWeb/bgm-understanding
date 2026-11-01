"""Build a bounded editorial interface with complete inline timing candidates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA = 'bgm-editing-brief-v2'


def sound_motion(document):
    """Retain the measured envelope, rather than only section-wide averages.

    Detailed reports already bound the duration-weighted envelope to 48 windows.
    Use band power shares (not instrument identities) to expose bass/brightness shifts.
    """
    rows = document['dynamics'].get('windows', [])
    if len(rows) > 48:
        duration = document['audio']['duration_seconds']
        aggregated = []
        for i in range(48):
            start, end = i * duration / 48, (i + 1) * duration / 48
            band_levels = {}
            for name in ('low', 'mid', 'high'):
                contributors = [(max(0, min(end, r['end']) - max(start, r['start'])),
                                 r.get('band_rms', {}).get(name)) for r in rows]
                contributors = [(w, v) for w, v in contributors if w > 0 and v is not None]
                if contributors:
                    band_levels[name] = math.sqrt(sum(w * v * v for w, v in contributors) /
                                                  sum(w for w, _ in contributors))
            aggregated.append({'start': start, 'end': end, 'band_rms': band_levels,
                               'relative_energy': weighted(rows, start, end, 'relative_energy'),
                               'detected_onsets_per_second': weighted(rows, start, end, 'detected_onsets_per_second')})
        rows = aggregated
    points = []
    for row in rows:
        bands = row.get('band_rms', {})
        powers = [max(0, bands.get(name, 0)) ** 2 for name in ('low', 'mid', 'high')]
        total = sum(powers)
        shares = [round(value / total, 3) for value in powers] if total else [None] * 3
        points.append([round(row['start'], 3), round(row['end'], 3),
                       round(row.get('relative_energy', 0), 3), *shares,
                       round(row.get('detected_onsets_per_second', 0), 2)])
    return {'columns': ['start', 'end', 'energy', 'low_share', 'mid_share', 'high_share', 'onsets_per_second'],
            'windows': points,
            'meaning': 'Duration-weighted envelope, at most 48 windows; energy is relative within this audio, band shares are power proportions, null means unavailable. Not instrument identity or emotion; use onsets for exact attacks.'}


def check_editorial_contract(document):
    """Bound descriptive overhead without deleting genuine beat/onset timing."""
    allowed = {'schema', 'status', 'audio', 'overview', 'timeline', 'change_candidates',
               'sound_motion', 'cut_points', 'text_cues', 'text_status', 'reading_notes', 'cautions', 'detail_ref'}
    errors = []
    if set(document) - allowed:
        errors.append('Unexpected main-output fields: raw diagnostics belong in the detail file.')
    for field, maximum in [('timeline', 96), ('change_candidates', 24)]:
        if len(document.get(field, [])) > maximum:
            errors.append(f'{field} exceeds its editorial row budget ({maximum}).')
    if len(document.get('sound_motion', {}).get('windows', [])) > 48:
        errors.append('Sound envelope exceeds 48 windows.')
    narrative = {key: value for key, value in document.items() if key not in ('cut_points', 'text_cues')}
    narrative_bytes = len(json.dumps(narrative, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    if narrative_bytes > 28000:
        errors.append('Descriptive overhead exceeds 28 KB; inspect duplicated diagnostics.')
    if document.get('text_status') == 'skipped_not_requested' and document.get('text_cues'):
        errors.append('Disabled transcription must not contain text cues.')
    return {'valid': not errors, 'errors': errors, 'narrative_bytes': narrative_bytes,
            'beat_count': len(document.get('cut_points', {}).get('beats', [])),
            'onset_count': len(document.get('cut_points', {}).get('onsets', []))}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def weighted(rows, start, end, key, default=0):
    pairs = [(max(0, min(end,r['end'])-max(start,r['start'])), r.get(key)) for r in rows]
    pairs = [(w,v) for w,v in pairs if w > 0 and isinstance(v,(int,float))]
    return sum(w*v for w,v in pairs)/sum(w for w,v in pairs) if pairs else default


def labels(rows, start, end, key='candidates', limit=3):
    totals = {}
    for row in rows:
        overlap = max(0, min(end,row['end'])-max(start,row['start']))
        for item in row.get(key, []):
            name = item.get('label')
            if name and overlap:
                totals[name] = totals.get(name,0)+overlap*item.get('score',0)
    return sorted(totals,key=totals.get,reverse=True)[:limit]


def cut_points(detail_path,duration):
    from extract_bgm_events import extract
    events=extract(detail_path,0,duration)
    return {'beats':[round(v,3) for v in events['beats']],
            'bar_starts':[round(v,3) for v in events['bar_starts']],
            'onsets':[[round(v['seconds'],3),round(v.get('strength',v.get('relative_strength',0)) or 0,4)] for v in events['onsets']],
            'onset_columns':['seconds','relative_strength'],
            'onset_coverage':events.get('onset_coverage','summary_only')}


def summarize(document, detail_path):
    duration = document['audio']['duration_seconds']
    speech = document['rhythm'].get('applicability') == 'low_spoken_audio'
    protected = [v['change_seconds'] for v in document.get('entry_candidates',[])] if not speech else []
    sections = []
    for row in document['structure'].get('sections', []):
        label = row.get('label_candidate')
        if (sections and sections[-1]['section_candidate'] == label
                and abs(sections[-1]['end']-row['start']) < .05
                and not any(abs(row['start']-point)<.05 for point in protected)):
            sections[-1]['end'] = row['end']
        else:
            sections.append({'start':row['start'],'end':row['end'],'section_candidate':label})
    if not sections:
        sections=[{'start':0,'end':duration,'section_candidate':None}]
    # A meaningful change can occur inside a model section or between equal labels.
    divided=[]
    for section in sections:
        edges=[section['start']]+sorted(set(p for p in protected if section['start']+.05<p<section['end']-.05))+[section['end']]
        divided += [{'start':a,'end':b,'section_candidate':section['section_candidate']} for a,b in zip(edges,edges[1:])]
    sections=divided
    energy = document['dynamics'].get('windows',[])
    voice = document['vocals'].get('timeline',[])
    sounds = document['instruments'].get('timeline',[])
    moods = document['emotion'].get('timeline',[])
    timeline=[]
    for row in sections:
        start,end=row['start'],row['end']
        third=(end-start)/3
        early=weighted(energy,start,start+third,'relative_energy')
        late=weighted(energy,end-third,end,'relative_energy')
        item={'start':round(start,2),'end':round(end,2),'section_candidate':row['section_candidate'],
              'energy':round(weighted(energy,start,end,'relative_energy'),2),
              'energy_trend':'rising' if late-early>.15 else 'falling' if early-late>.15 else 'steady',
              'onsets_per_second':round(weighted(energy,start,end,'detected_onsets_per_second'),1)}
        names=labels(sounds,start,end)
        if names:item['sound_candidates']=names
        if speech:item['voice']='speech'
        elif weighted(voice,start,end,'vocal_score')>=.3:item['voice']='possible_vocals'
        else:item['voice']='no_clear_vocal_evidence'
        if not speech:
            scores={}
            for m in moods:
                for key,value in m.get('scores',{}).items():
                    overlap=max(0,min(end,m['end'])-max(start,m['start']))
                    scores[key]=scores.get(key,0)+overlap*value/(end-start)
            best=[key for key in sorted(scores,key=scores.get,reverse=True) if scores[key]>=.55][:2]
            if best:item['mood_candidates']=best
        timeline.append(item)
    overview={'bpm_candidate':None if speech else document['rhythm'].get('bpm_candidate'),
              'main_sound_candidates':[r['label'] for r in document['instruments'].get('summary',[])[:3]],
              'audio_kind':'speech_dominant' if speech else 'music',
              'language_candidate':document['lyrics'].get('language') if document['lyrics'].get('segments') else None}
    bars=document['rhythm'].get('bar_starts_seconds',[])
    if len(bars)>1 and not speech:
        overview['typical_bar_seconds']=round(statistics.median(b-a for a,b in zip(bars,bars[1:])),3)
    if not speech and document['rhythm'].get('grid_source'):
        overview['rhythm_source']=document['rhythm']['grid_source']
    if speech:
        summary=document['emotion'].get('speech_emotion',{}).get('summary')
        if summary:overview['speech_attribute_candidates']={k:round(v,2) for k,v in summary.items()}
    else:
        scores=document['emotion'].get('summary_scores',{})
        overview['mood_candidates']=[k for k in sorted(scores,key=scores.get,reverse=True) if scores[k]>=.55][:2]
    role=document['lyrics'].get('content_role_candidate',{}).get('label')
    if role:overview['text_role_candidate']=role
    cues=[]
    for row in document.get('entry_candidates',[]):
        item={'time':round(row['change_seconds'],2),'type':'change_candidate'}
        signals=set()
        for evidence in row.get('evidence',[]):
            signals.update(evidence.get('types',evidence.get('signals',[])))
            if evidence.get('source')=='allin1':signals.add('section_boundary_candidate')
        if signals:item['signals']=sorted(signals)
        if row.get('aligned_bar_start_seconds') is not None:item['nearby_bar_start']=round(row['aligned_bar_start_seconds'],2)
        cues.append(item)
    text=[]
    for row in document['lyrics'].get('segments',[]):
        item={'start':round(row['start'],2),'end':round(row['end'],2),
              'text':row.get('normalized_text_candidate',row['text']),
              'source':'context_correction_candidate' if row.get('normalized_text_candidate') else 'asr_candidate'}
        text.append(item)
    cautions=['Musical labels and text are unverified candidates; text may be incomplete.']
    if document['vocals'].get('model_disagreement',{}).get('flagged_windows',0)>0:
        cautions.append('Voice classifiers disagree; possible_vocals does not confirm singing.')
    if speech:
        cautions.append('Use phrase timing, not an inferred musical grid; speech attributes are not validated for Mandarin.')
    result = {'schema':SCHEMA,'status':document['status'],
            'audio':{'name':Path(document['audio']['original_file']).name,'duration_seconds':round(duration,3),
                     'source_sha256':document['audio'].get('original_sha256'),
                     'time_origin':'decoded audio start','source_start_seconds':document['audio']['source_start_seconds']},
            'overview':overview,'timeline':timeline,'change_candidates':sorted(cues,key=lambda r:r['time']),
            'sound_motion':sound_motion(document),
            'cut_points':cut_points(detail_path,duration),'text_cues':text,
            'reading_notes':{'energy':'Relative level within this audio, 0–1; not emotion.',
                             'onsets_per_second':'Detected sound density; not BPM.',
                             'cut_points':'Seconds from audio start: beats=beat grid; bar_starts=bar-first-beat candidates; onsets=[time,strength] detected attacks. These are options, not mandatory cuts or instrument identities.',
                             'text_cues':'Phrase candidates, not corrected subtitles. Empty means no supported words, not confirmed absence of vocals.'},
            'cautions':cautions,
            'detail_ref':{'file':str(Path(detail_path).resolve()),'sha256':digest(detail_path),
                          'use':'Cut-point candidates are already included above. Read details only for model diagnostics, alternative text or full-resolution measurements.'}}
    if document['lyrics'].get('status') == 'skipped_not_requested':
        result['text_status'] = 'skipped_not_requested'
        result['reading_notes']['text_cues'] = 'Transcription was not requested. Empty text does not indicate absence of vocals.'
        result['cautions'][0] = 'Musical labels are unverified model candidates.'
    return result


def write_brief(path, document):
    contract = check_editorial_contract(document)
    if not contract['valid']:
        raise ValueError('Editorial contract failed: ' + '; '.join(contract['errors']))
    # Real reduction happens in summarize(); one timeline/phrase per line also makes scanning easier.
    lines=['{']
    entries=list(document.items())
    for i,(key,value) in enumerate(entries):
        prefix='  '+json.dumps(key)+': '
        if key in ('cut_points','sound_motion'):
            lines.append(prefix+'{')
            items=list(value.items())
            for j,(name,values) in enumerate(items):
                lines.append('    '+json.dumps(name)+': '+json.dumps(values,ensure_ascii=False,separators=(',',':'),allow_nan=False)+(',' if j<len(items)-1 else ''))
            lines.append('  }'+(',' if i<len(entries)-1 else ''))
        elif isinstance(value,list) and value and all(isinstance(v,dict) for v in value):
            lines.append(prefix+'[')
            lines += ['    '+json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)+(',' if j<len(value)-1 else '') for j,v in enumerate(value)]
            lines.append('  ]'+(',' if i<len(entries)-1 else ''))
        else:
            parts=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).splitlines()
            lines.append(prefix+parts[0])
            lines += ['  '+part for part in parts[1:]]
            if i<len(entries)-1:lines[-1]+=','
    lines.append('}')
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def validate_brief(document,expected_audio=None,require_enriched=False,decode_source=False):
    from validate_agent_json import validate
    errors=[]
    reference=document.get('detail_ref',{})
    path=Path(reference.get('file',''))
    if not path.is_file() or digest(path)!=reference.get('sha256'):
        return {'valid':False,'errors':['Detailed evidence missing or changed.'],'warnings':[],'checks':[],'check_count':0}
    detail=json.loads(path.read_text(encoding='utf-8-sig'))
    if detail.get('schema') in [SCHEMA,'bgm-editing-brief-v1']:
        return {'valid':False,'errors':['detail_ref must point to analysis, not another brief.'],'warnings':[],'checks':[],'check_count':0}
    result=validate(detail,expected_audio,require_enriched,decode_source)
    errors.extend(check_editorial_contract(document)['errors'])
    duration=detail['audio']['duration_seconds']
    brief_duration=document.get('audio',{}).get('duration_seconds',-1)
    if not isinstance(brief_duration,(int,float)) or not math.isfinite(brief_duration) or abs(brief_duration-duration)>.025:
        errors.append('Brief duration differs from evidence.')
    if document.get('audio',{}).get('source_sha256')!=detail['audio'].get('original_sha256'):
        errors.append('Brief audio identity differs from evidence.')
    if document.get('status')!=detail.get('status'):errors.append('Brief status differs from evidence.')
    if document.get('schema')==SCHEMA:
        expected_points=cut_points(path,duration)
        if document.get('cut_points')!=expected_points:
            errors.append('Cut points missing or changed from evidence.')
        if 'sound_motion' in document and document['sound_motion'] != sound_motion(detail):
            errors.append('Sound envelope differs from measured evidence.')
    last=0
    for row in document.get('timeline',[]):
        if not all(isinstance(row.get(k),(float,int)) and math.isfinite(row[k]) for k in ['start','end']):
            errors.append('Nonfinite brief interval.');continue
        if abs(row['start']-last)>.025 or not 0<=row['start']<row['end']<=duration+.025:
            errors.append('Brief timeline does not partition the analyzed audio.')
        last=row['end']
    if abs(last-duration)>.025:errors.append('Brief timeline misses the audio tail.')
    for cue in document.get('change_candidates',[]):
        if not isinstance(cue.get('time'),(int,float)) or not math.isfinite(cue['time']) or not 0<=cue['time']<=duration:
            errors.append('Invalid brief change point.')
    original=detail['lyrics'].get('segments',[])
    cues=document.get('text_cues',[])
    if len(cues)!=len(original):errors.append('Primary text cues were lost.')
    for row,source in zip(cues,original):
        if (abs(row['start']-source['start'])>.025 or abs(row['end']-source['end'])>.025
                or row['text']!=source.get('normalized_text_candidate',source['text'])):
            errors.append('Brief text or phrase timing changed without evidence.')
    result['errors']+=errors
    result['checks']+=['Brief is bound to detailed evidence by SHA256.','Brief covers analyzed duration and preserves primary phrase cues.']
    result['check_count']=len(result['checks'])
    result['valid']=not result['errors']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; use a new file.')
    data=json.loads(a.analysis.read_text(encoding='utf-8-sig'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_brief(a.output,summarize(data,a.analysis))
    print(json.dumps({'output':str(a.output),'bytes':a.output.stat().st_size}))


if __name__=='__main__':main()
