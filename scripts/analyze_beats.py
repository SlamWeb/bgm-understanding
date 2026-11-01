"""Run locally installed Beat This on a PCM WAV; timestamps use WAV seconds."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.io import wavfile

p=argparse.ArgumentParser()
p.add_argument('wav',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--model',type=Path)
p.add_argument('--runtime-root',type=Path,default=Path(os.environ.get('BGM_ANALYSIS_ROOT',Path(__file__).resolve().parents[1]/'.runtime')))
p.add_argument('--threads',type=int,default=2)
args=p.parse_args()
args.model=args.model or args.runtime_root/'models/small0.ckpt'
if not args.model.is_file(): p.error('Cached Beat This checkpoint missing; run setup_runtime.py.')
sys.path.insert(0,str(args.runtime_root/'packages'))
from beat_this.inference import Audio2Beats
if args.output.exists(): raise FileExistsError('Choose a new output file.')
import torch
torch.set_num_threads(args.threads)
sr, samples=wavfile.read(args.wav)
if np.issubdtype(samples.dtype,np.integer):
    samples=samples.astype(np.float32)/max(abs(np.iinfo(samples.dtype).min),np.iinfo(samples.dtype).max)
elif np.issubdtype(samples.dtype,np.floating):
    samples=samples.astype(np.float32)
else: raise ValueError('Unsupported WAV sample type')
start=time.perf_counter()
detector=Audio2Beats(checkpoint_path=str(args.model.resolve()),device='cpu',dbn=False)
beats,downbeats=detector(samples,sr)
result={'schemaVersion':1,'type':'music_beat_prediction','sourceAudio':str(args.wav.resolve()),
    'audioSha256':hashlib.sha256(args.wav.read_bytes()).hexdigest(),'timeOrigin':'input WAV start',
    'durationSeconds':round(len(samples)/sr,6),'sampleRate':sr,
    'model':'Beat This small0','modelSha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
    'device':'cpu','durationProcessingSeconds':round(time.perf_counter()-start,2),
    'beatSeconds':[round(float(x),4) for x in beats],
    'downbeatSeconds':[round(float(x),4) for x in downbeats],
    'humanListeningVerified':False,
    'limitation':'Predicted beats/downbeats are not instrument onsets; no instrument classification.'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
