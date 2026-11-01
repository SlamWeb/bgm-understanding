"""Negative contract tests for the enriched BGM handoff, without models."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

from validate_agent_json import validate
from enrich_understanding import build as enrich


class EnrichedContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / 'source.wav'
        with wave.open(str(path), 'wb') as w:
            w.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
            w.writeframes(b'\0' * 44100 * 4 * 2)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        module = {'status': 'complete', 'method': 'test semantic model',
                  'timeline': [{'start': 0, 'end': 2, 'candidates': []}]}
        self.document = {
            'schema': 'bgm-understanding-v2', 'status': 'complete',
            'audio': {'file': str(path), 'original_file': str(path), 'sha256': digest,
                      'original_sha256': digest, 'duration_seconds': 2,
                      'source_start_seconds': 0, 'time_origin': 'canonical_wav_start'},
            'rhythm': {'beats_seconds': [.1, .6, 1.1, 1.6], 'bar_starts_seconds': [.1]},
            'sound_events': {'events': []},
            'dynamics': {'windows': [{'start': 0, 'end': 1}, {'start': 1, 'end': 2}]},
            'structure': {'sections': [{'start': 0, 'end': 2, 'label_candidate': 'verse'}]},
            'entry_candidates': [], 'uncertainty': {'human_listened': False},
            'instruments': copy.deepcopy(module), 'vocals': copy.deepcopy(module),
            'emotion': copy.deepcopy(module),
            'lyrics': {'status': 'complete', 'model': 'test-asr',
                       'segments': [{'start': .2, 'end': 1.8, 'text': 'example',
                                     'words': [{'start': .2, 'end': 1.8, 'text': 'example'}]}]},
        }

    def result(self, **kwargs):
        return validate(self.document, require_enriched=True, **kwargs)

    def test_valid_whole_source_and_enriched_document(self):
        result = self.result(decode_source=True)
        self.assertTrue(result['valid'], result['errors'])
        self.assertIn('whole original audio is covered', result['checks'])

    def test_intentionally_skipped_transcription_is_not_analysis_failure(self):
        self.document['lyrics']={'status':'skipped_not_requested','requested':False,'segments':[]}
        result=self.result()
        self.assertTrue(result['valid'],result['errors'])
        self.assertFalse(any('no retained text' in x for x in result['warnings']))
        self.document['lyrics']['segments']=[{'start':0,'end':1,'text':'should not be here'}]
        self.assertFalse(self.result()['valid'])

    def test_end_time_outside_audio_is_rejected(self):
        self.document['lyrics']['segments'][0]['end'] = 2.3
        self.assertFalse(self.result()['valid'])

    def test_gap_at_tail_is_rejected(self):
        self.document['dynamics']['windows'][-1]['end'] = 1.7
        self.assertFalse(self.result()['valid'])

    def test_original_audio_changed_is_rejected(self):
        self.document['audio']['original_sha256'] = '0' * 64
        self.assertFalse(self.result()['valid'])

    def test_missing_semantic_capability_is_rejected(self):
        del self.document['emotion']
        self.assertFalse(self.result()['valid'])

    def test_energy_only_emotion_is_rejected(self):
        self.document['emotion']['method'] = 'rms energy threshold'
        self.assertFalse(self.result()['valid'])

    def test_instrumental_empty_lyrics_is_valid_with_warning(self):
        self.document['lyrics']['segments'] = []
        self.document['lyrics']['status'] = 'no_reliable_lyrics'
        result = self.result()
        self.assertTrue(result['valid'], result['errors'])
        self.assertTrue(any('no retained text' in x for x in result['warnings']))

    def test_nan_timeline_is_rejected(self):
        self.document['vocals']['timeline'][0]['start'] = float('nan')
        self.assertFalse(self.result()['valid'])

    def test_nonfinite_model_score_is_rejected(self):
        self.document['instruments']['timeline'][0]['candidates'] = [{'label': 'piano', 'score': float('nan')}]
        self.assertFalse(self.result()['valid'])

    def test_complete_status_cannot_hide_failed_module(self):
        self.document['emotion']['status'] = 'failed'
        self.assertFalse(self.result()['valid'])


class EnrichmentAssemblyTests(unittest.TestCase):
    """The assembly boundary must reject reports measured on different audio."""

    setUp = EnrichedContractTests.setUp

    def write_reports(self):
        root = Path(self.tmp.name)
        core = {k: copy.deepcopy(v) for k, v in self.document.items()
                if k not in ('instruments', 'vocals', 'emotion', 'lyrics')}
        core.update(editing_use={}, raw_files={})
        audio = core['audio']
        source = {'source': audio['file'], 'sha256': audio['sha256'],
                  'duration_seconds': audio['duration_seconds'], 'time_origin': 'canonical_wav_start'}
        semantics = {'audio': copy.deepcopy(source), 'models': {'model': 'fixture'},
                     **{k: copy.deepcopy(self.document[k]) for k in ('instruments', 'vocals', 'emotion')}}
        lyrics = {'audio': copy.deepcopy(source), **copy.deepcopy(self.document['lyrics']),
                  'raw_segments': [], 'rejected_segments': [], 'human_reviewed': False}
        for name, value in [('core', core), ('semantics', semantics), ('lyrics', lyrics)]:
            (root / (name + '.json')).write_text(json.dumps(value), encoding='utf-8')
        return root, core, semantics, lyrics

    def rewrite(self, root, name, value):
        (root / (name + '.json')).write_text(json.dumps(value), encoding='utf-8')

    def assembled(self, root):
        return enrich(root / 'core.json', root / 'semantics.json', root / 'lyrics.json')

    def test_same_audio_reports_assemble_without_claiming_human_review(self):
        root, _, _, _ = self.write_reports()
        result = self.assembled(root)
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['uncertainty']['human_listened'])
        self.assertNotIn('raw_segments', result['lyrics'])
        self.assertIn('raw_report_file', result['lyrics'])

    def test_different_audio_semantics_is_rejected(self):
        root, _, semantic, _ = self.write_reports()
        semantic['audio']['sha256'] = '0' * 64
        self.rewrite(root, 'semantics', semantic)
        with self.assertRaises(ValueError):
            self.assembled(root)

    def test_nonfinite_transcript_interval_is_rejected(self):
        root, _, _, lyrics = self.write_reports()
        lyrics['segments'][0]['start'] = float('nan')
        self.rewrite(root, 'lyrics', lyrics)
        with self.assertRaises(ValueError):
            self.assembled(root)

    def test_outside_transcript_interval_is_rejected(self):
        root, _, _, lyrics = self.write_reports()
        lyrics['segments'][0]['end'] = 2.5
        self.rewrite(root, 'lyrics', lyrics)
        with self.assertRaises(ValueError):
            self.assembled(root)

    def test_failed_requested_module_produces_partial(self):
        root, _, semantic, _ = self.write_reports()
        semantic['emotion']['status'] = 'failed'
        self.rewrite(root, 'semantics', semantic)
        result = self.assembled(root)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['analysis_status']['emotion'], 'failed')

    def test_nonfinite_report_coverage_is_rejected(self):
        root, _, semantic, _ = self.write_reports()
        semantic['audio']['duration_seconds'] = float('nan')
        self.rewrite(root, 'semantics', semantic)
        with self.assertRaises(ValueError):
            self.assembled(root)

    def test_different_report_time_origin_is_rejected(self):
        root, _, _, lyrics = self.write_reports()
        lyrics['audio']['time_origin'] = 'original_file_start_plus_unknown_offset'
        self.rewrite(root, 'lyrics', lyrics)
        with self.assertRaises(ValueError):
            self.assembled(root)


if __name__ == '__main__':
    unittest.main()
