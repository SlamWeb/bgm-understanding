"""The editing interface retains meaning/times, without dense measurements."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from summarize_bgm import summarize,write_brief,check_editorial_contract,sound_motion
from extract_bgm_events import extract


class BriefTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'detail.json'
        self.data={'status':'complete','audio':{'original_file':'sample.mp3','duration_seconds':4,
                    'original_sha256':'source-hash','source_start_seconds':0,'time_origin':'decoded audio start'},
            'rhythm':{'bpm_candidate':120,'beats_seconds':[0,.5,1,1.5,2,2.5,3,3.5],'bar_starts_seconds':[0,2]},
            'structure':{'sections':[{'start':0,'end':2,'label_candidate':'intro'},
                                     {'start':2,'end':4,'label_candidate':'intro'}]},
            'dynamics':{'windows':[{'start':0,'end':2,'relative_energy':.2,'detected_onsets_per_second':1},
                                   {'start':2,'end':4,'relative_energy':.8,'detected_onsets_per_second':3}]},
            'sound_events':{'events':[{'seconds':.75,'relative_strength':.8}]},
            'instruments':{'summary':[{'label':'piano'}],'timeline':[{'start':0,'end':4,'candidates':[{'label':'piano','score':.8}]}]},
            'vocals':{'timeline':[],'model_disagreement':{'flagged_windows':0}},
            'emotion':{'summary_scores':{'relaxed':.8},'timeline':[]},
            'lyrics':{'language':'en','segments':[{'start':1,'end':2,'text':'raw phrase','normalized_text_candidate':'candidate phrase'}]},
            'entry_candidates':[{'change_seconds':2,'aligned_bar_start_seconds':2}]}
        self.path.write_text(json.dumps(self.data),encoding='utf-8')

    def test_projection_preserves_coverage_and_text_status(self):
        before=copy.deepcopy(self.data)
        result=summarize(self.data,self.path)
        self.assertEqual(result['timeline'][0]['start'],0)
        self.assertEqual(result['timeline'][-1]['end'],4)
        self.assertEqual(len(result['timeline']),2)
        self.assertEqual(result['text_cues'][0]['text'],'candidate phrase')
        self.assertEqual(result['text_cues'][0]['source'],'context_correction_candidate')
        self.assertEqual(result['cut_points']['beats'],self.data['rhythm']['beats_seconds'])
        self.assertEqual(result['cut_points']['bar_starts'],[0,2])
        self.assertEqual(result['cut_points']['onsets'],[[.75,.8]])
        self.assertEqual(before,self.data)
        self.assertFalse(any('disagree' in text for text in result['cautions']))

    def test_equal_labels_merge_only_when_no_important_change(self):
        self.data['entry_candidates']=[]
        self.assertEqual(len(summarize(self.data,self.path)['timeline']),1)

    def test_unreadable_vocals_do_not_establish_language(self):
        self.data['lyrics']['segments']=[]
        self.assertIsNone(summarize(self.data,self.path)['overview']['language_candidate'])

    def test_skipped_transcription_preserves_cut_points_without_empty_lyrics_warning(self):
        self.data['lyrics']={'status':'skipped_not_requested','requested':False,'segments':[]}
        result=summarize(self.data,self.path)
        self.assertEqual(result['text_status'],'skipped_not_requested')
        self.assertEqual(result['text_cues'],[])
        self.assertEqual(result['cut_points']['beats'],self.data['rhythm']['beats_seconds'])
        self.assertIn('not requested',result['reading_notes']['text_cues'])

    def test_readable_serialization_is_valid_json(self):
        brief=summarize(self.data,self.path)
        path=Path(self.tmp.name)/'brief.json'
        write_brief(path,brief)
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')),brief)

    def test_envelope_exposes_bass_and_energy_rise(self):
        self.data['dynamics']['windows'][0]['band_rms']={'low':1,'mid':2,'high':0}
        self.data['dynamics']['windows'][1]['band_rms']={'low':3,'mid':1,'high':0}
        motion=sound_motion(self.data)
        self.assertEqual(motion['windows'][0],[0,2,.2,.2,.8,0,1])
        self.assertEqual(motion['windows'][1],[2,4,.8,.9,.1,0,3])

    def test_long_envelope_is_bounded_but_reaches_tail(self):
        self.data['audio']['duration_seconds']=120
        self.data['dynamics']['windows']=[{'start':i,'end':i+1,'relative_energy':i/120,
            'band_rms':{'low':1,'mid':1,'high':1},'detected_onsets_per_second':2} for i in range(120)]
        rows=sound_motion(self.data)['windows']
        self.assertEqual(len(rows),48)
        self.assertEqual(rows[0][0],0)
        self.assertEqual(rows[-1][1],120)
        self.assertGreater(rows[-1][2],rows[0][2])

    def test_raw_dump_rejected_without_truncating_valid_event_arrays(self):
        brief=summarize(self.data,self.path)
        brief['cut_points']['onsets']=[[i/100, .5] for i in range(10000)]
        self.assertTrue(check_editorial_contract(brief)['valid'])
        brief['debug']={'scores':[0]*10000}
        self.assertFalse(check_editorial_contract(brief)['valid'])
        with self.assertRaises(ValueError):write_brief(Path(self.tmp.name)/'bad.json',brief)

    def test_repeated_narrative_is_not_hidden_by_single_line_serialization(self):
        brief=summarize(self.data,self.path)
        brief['reading_notes']['extra']='diagnostic ' * 4000
        self.assertFalse(check_editorial_contract(brief)['valid'])

    def test_exact_events_are_retrieved_for_selected_range(self):
        result=extract(self.path,.5,1.5)
        self.assertEqual(result['beats'],[.5,1])
        self.assertEqual(len(result['onsets']),1)
        with self.assertRaises(ValueError):extract(self.path,3,5)

    def test_cut_points_still_work_without_detailed_file(self):
        brief=summarize(self.data,self.path)
        path=Path(self.tmp.name)/'brief.json'
        write_brief(path,brief)
        self.path.unlink()
        self.assertEqual(extract(path,.5,1.5)['beats'],[.5,1])
        self.assertEqual(extract(path,.5,1.5)['onsets'][0]['seconds'],.75)

    def test_legacy_changed_detail_is_rejected(self):
        brief=summarize(self.data,self.path)
        brief['schema']='bgm-editing-brief-v1'
        brief.pop('cut_points')
        path=Path(self.tmp.name)/'brief.json'
        write_brief(path,brief)
        self.path.write_text('{}',encoding='utf-8')
        with self.assertRaises(ValueError):extract(path,0,1)


if __name__=='__main__':unittest.main()
