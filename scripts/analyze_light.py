"""Sonara signal-based analysis; retain raw values and expose frame times in seconds."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('audio',type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--runtime-root',type=Path,default=Path(os.environ.get('BGM_ANALYSIS_ROOT',Path(__file__).resolve().parents[1]/'.runtime')))
args=parser.parse_args()
sys.path.insert(0,str(args.runtime_root/'packages'))
import sonara
if args.output.exists(): raise FileExistsError('Use a new output filename.')
features=['bpm','beats','onsets','rms','onset_density','beatgrid','structure','loudness','silence']
start=time.perf_counter()
raw=sonara.analyze_file(str(args.audio),features=features)
elapsed=time.perf_counter()-start
prov=raw['provenance']; scale=prov['hop_length']/prov['sample_rate']
def times(frames): return [round(float(f)*scale,4) for f in frames]
result={'schemaVersion':1,'audio':{'source':str(args.audio.resolve()),
  'sha256':hashlib.sha256(args.audio.read_bytes()).hexdigest(),
  'durationSeconds':float(raw['duration_sec']),'timeOrigin':'input file start'},
  'engine':{'name':'sonara','version':sonara.__version__,'device':'cpu',
  'processingSeconds':round(elapsed,3),'provenance':prov},
  'rhythm':{'bpmCandidate':float(raw['bpm']),'beatsSeconds':times(raw['beats']),
  'downbeatsSeconds':times(raw['downbeats']),
  'downbeatNote':'Grid phase estimated by accent; assumes 4/4 unless time signature requested.'},
  'onsets':{'candidateSeconds':times(raw['onset_frames']),
  'meanDensityPerSecond':float(raw['onset_density']),'instrumentLabels':None},
  'dynamics':{'energyCurve':raw['energy_curve'],
  'energyCurveHopSeconds':raw['energy_curve_hop_sec'],'loudnessCurve':raw['loudness_curve'],
  'loudnessNote':'Short-term LUFS: 3-second window, 1-second hop; verify alignment before use.'},
  'segments':raw['segments'],
  'limitations':['Analysis is unreviewed; beat, onset and boundaries are estimates.',
  'Sections use minimum 8-second spacing and may miss changes in short clips.',
  'No instrument classification; energy scores are normalized heuristic values.',
  'Rhythm may be detected at half or double tempo.'],
  'raw':dict(raw)}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'output':str(args.output.resolve()),'seconds':result['audio']['durationSeconds'],
 'beats':len(result['rhythm']['beatsSeconds']),'onsets':len(result['onsets']['candidateSeconds']),
 'segments':len(result['segments']),'processingSeconds':round(elapsed,3)},ensure_ascii=False))
