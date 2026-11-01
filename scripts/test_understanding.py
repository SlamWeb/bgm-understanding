"""Handoff integrity and time-contract tests; synthetic reports are not accuracy evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave
from build_understanding import build,grid_tempo,select_grid


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.wav=self.root/'audio.wav'
        with wave.open(str(self.wav),'wb') as w:
            w.setparams((1,2,8000,0,'NONE','not compressed'))
            w.writeframes(b'\0\0'*32000)
        self.sha=hashlib.sha256(self.wav.read_bytes()).hexdigest()
        self.write('onsets.json',{'sourceAudio':str(self.wav),'analysisOffset':0,'events':[]})
        self.write('analysis.json',{'audio':{'sha256':self.sha,'durationSeconds':4},
            'sources':{'onsets':{'path':str(self.root/'onsets.json'),'sha256':hashlib.sha256((self.root/'onsets.json').read_bytes()).hexdigest()}},
            'dynamics':{'waveformLevel':{'nearSilent':False},'summaryWindows':[]},
            'rhythm':{'beatSeconds':[0,1,2,3], 'downbeatSeconds':[0,2],
                       'tempo':{'beatThisMedianIntervalBpm':60,'sonaraBpmCandidate':60,'relation':'approximately_agree'}},
            'structure':{'longSectionCandidates':[{'rangeSeconds':[0,4]}],
                         'shortWindowChangeCandidates':[{'timeSeconds':1.5,'signals':['energy_increase']}]}})
        self.write('receipt.json',{'canonicalWav':str(self.wav),'inputFile':str(self.wav),'inputSha256':self.sha,'sourceStartSeconds':12})
        self.write('features.json',{'audio':{'source':str(self.wav),'sha256':self.sha,'duration_seconds':4},
            'windows':[{'start':0,'end':4,'rms':.1,'dbfs':-20,'band_rms':{'low':.1,'mid':0,'high':0}}],
            'changes':[{'seconds':1.5,'types':['energy_increase']}]})

    def write(self,name,data):
        (self.root/name).write_text(json.dumps(data),encoding='utf-8')

    def read(self,name):
        return json.loads((self.root/name).read_text(encoding='utf-8'))

    def result(self):
        return build(self.root/'analysis.json',self.root/'receipt.json',self.root/'features.json')

    def test_partial_and_source_offset_preserved_once(self):
        d=self.result()
        self.assertEqual(d['status'],'partial')
        self.assertEqual(d['audio']['source_start_seconds'],12)
        self.assertEqual(d['rhythm']['beats_seconds'],[0,1,2,3])
        self.assertEqual(d['entry_candidates'][0]['aligned_bar_start_seconds'],2)
        self.assertFalse(d['uncertainty']['human_listened'])

    def test_silence_suppresses_model_predictions(self):
        a=self.read('analysis.json');a['dynamics']['waveformLevel']['nearSilent']=True;self.write('analysis.json',a)
        d=self.result()
        self.assertEqual(d['rhythm']['beats_seconds'],[])
        self.assertEqual(d['entry_candidates'],[])
        self.assertIsNone(d['rhythm']['bpm_candidate'])

    def test_mismatched_audio_rejected(self):
        f=self.read('features.json');f['audio']['sha256']='wrong';self.write('features.json',f)
        with self.assertRaises(ValueError):self.result()

    def test_far_outside_event_rejected(self):
        a=self.read('analysis.json');a['rhythm']['beatSeconds']=[9];self.write('analysis.json',a)
        with self.assertRaises(ValueError):self.result()

    def test_modified_event_report_rejected(self):
        self.write('onsets.json',{'sourceAudio':str(self.wav),'analysisOffset':0,'events':[{'accepted':True,'timeSeconds':2,'strength':1}]})
        with self.assertRaises(ValueError):self.result()

    def test_structure_bookkeeping_edges_cover_whole_audio(self):
        folder=self.root/'structure';folder.mkdir()
        raw={'bpm':60,'beats':[0,1,2,3],'downbeats':[0,2],
             'segments':[{'start':0,'end':.06,'label':'start'},
                         {'start':.06,'end':2,'label':'verse'},
                         {'start':2,'end':3.98,'label':'chorus'},
                         {'start':3.98,'end':4.02,'label':'end'}]}
        self.write('structure/structure.json',raw)
        self.write('structure/receipt.json',{'sha256':self.sha,'duration_seconds':4,
                    'raw_sha256':hashlib.sha256((folder/'structure.json').read_bytes()).hexdigest()})
        d=build(self.root/'analysis.json',self.root/'receipt.json',self.root/'features.json',folder)
        self.assertEqual(d['status'],'complete')
        self.assertEqual(d['structure']['sections'][0]['start'],0)
        self.assertEqual(d['structure']['sections'][-1]['end'],4)
        self.assertEqual([b['seconds'] for b in d['structure']['boundaries']],[2])

    def test_structure_tracking_late_does_not_remove_earlier_beats(self):
        folder=self.root/'structure';folder.mkdir()
        raw={'bpm':60,'beats':[2,3],'downbeats':[2],
             'segments':[{'start':0,'end':4,'label':'verse'}]}
        self.write('structure/structure.json',raw)
        self.write('structure/receipt.json',{'sha256':self.sha,'duration_seconds':4,
                    'raw_sha256':hashlib.sha256((folder/'structure.json').read_bytes()).hexdigest()})
        d=build(self.root/'analysis.json',self.root/'receipt.json',self.root/'features.json',folder)
        self.assertEqual(d['rhythm']['beats_seconds'],[0,1,2,3])
        self.assertEqual(d['rhythm']['bar_starts_seconds'],[0,2])
        self.assertEqual(d['rhythm']['grid_source'],'Beat This')
        self.assertEqual(d['rhythm']['bpm_candidate'],60)

    def test_grid_selection_does_not_mix_half_double_tempo_tracks(self):
        beats,bars,source,_=select_grid({'beatSeconds':[i*.25 for i in range(16)],'downbeatSeconds':[0,1,2,3]},
            {'beats':[1,1.5,2,2.5,3,3.5],'downbeats':[1,3]},4)
        self.assertEqual(beats,[1,1.5,2,2.5,3,3.5])
        self.assertEqual(bars,[1,3])
        self.assertEqual(source,'All-In-One')

    def test_multibeat_tempo_reduces_frame_quantization(self):
        self.assertAlmostEqual(grid_tempo([0,.46,.94,1.4,1.88]),127.66,places=2)


if __name__=='__main__':unittest.main()
