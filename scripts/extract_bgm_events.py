"""Read exact cut-point candidates for a chosen range without loading all diagnostics into context."""
import argparse
import json
import hashlib
from pathlib import Path


def extract(detail,start,end):
    data=json.loads(Path(detail).read_text(encoding='utf-8-sig'))
    if data.get('schema')=='bgm-editing-brief-v2':
        duration=data['audio']['duration_seconds']
        if not 0<=start<end<=duration+.025:raise ValueError('Range must be inside analyzed audio.')
        points=data['cut_points']
        return {'range_seconds':[start,end],'time_origin':data['audio']['time_origin'],
                'beats':[v for v in points['beats'] if start<=v<end],
                'bar_starts':[v for v in points['bar_starts'] if start<=v<end],
                'onsets':[{'seconds':t,'strength':s} for t,s in points['onsets'] if start<=t<end],
                'onset_coverage':points.get('onset_coverage','unknown'),
                'meaning':'Point candidates, not mandatory cuts or identified instrument attacks.'}
    if data.get('schema')=='bgm-editing-brief-v1':
        linked=Path(data['detail_ref']['file'])
        if hashlib.sha256(linked.read_bytes()).hexdigest()!=data['detail_ref']['sha256']:
            raise ValueError('Detailed evidence changed.')
        data=json.loads(linked.read_text(encoding='utf-8-sig'))
    duration=data['audio']['duration_seconds']
    if not 0<=start<end<=duration+.025:raise ValueError('Range must be inside analyzed audio.')
    source=data['sound_events'].get('all_events_file')
    if source and Path(source).is_file():
        events=json.loads(Path(source).read_text(encoding='utf-8-sig')).get('events',[])
        onsets=[{'seconds':e['timeSeconds'],'strength':e.get('strength')} for e in events
                if e.get('accepted') and start<=e['timeSeconds']<end]
    else:
        onsets=[e for e in data['sound_events'].get('events',[]) if start<=e['seconds']<end]
    return {'range_seconds':[start,end],'time_origin':data['audio']['time_origin'],
            'beats':[v for v in data['rhythm']['beats_seconds'] if start<=v<end],
            'bar_starts':[v for v in data['rhythm']['bar_starts_seconds'] if start<=v<end],
            'onsets':onsets,'onset_coverage':'full_detected_events' if source and Path(source).is_file() else 'summary_only',
            'meaning':'Point candidates, not mandatory cuts or identified instrument attacks.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detail',type=Path,required=True)
    p.add_argument('--start',type=float,required=True)
    p.add_argument('--end',type=float,required=True)
    a=p.parse_args()
    print(json.dumps(extract(a.detail,a.start,a.end),ensure_ascii=False,separators=(',',':'),allow_nan=False))
