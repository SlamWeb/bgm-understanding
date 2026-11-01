"""Dense score summaries preserve time/peaks and report selection checks remain strict."""
import json
from pathlib import Path
import tempfile
import unittest

from enrich_understanding import compact_vocals, load_bound_report


class DenseHandoffTests(unittest.TestCase):
    def test_dense_summary_preserves_tail_peak_and_disagreement(self):
        rows = [{'start': i*.48, 'end': (i+1)*.48, 'vocal_score': .2,
                 'model_disagreement': False} for i in range(100)]
        rows[-1]['vocal_score'] = .9
        rows[-1]['model_disagreement'] = True
        activity = [{'start': 47.52, 'end': 48, 'type': 'uncertain_vocal_candidate'}]
        result = compact_vocals({'timeline': rows, 'segments': activity})
        self.assertLessEqual(len(result['timeline']), 64)
        self.assertEqual(result['timeline'][0]['start'], 0)
        self.assertEqual(result['timeline'][-1]['end'], 48)
        self.assertEqual(result['timeline'][-1]['vocal_score_peak'], .9)
        self.assertTrue(result['timeline'][-1]['model_disagreement'])
        self.assertEqual(result['segments'], activity)
        self.assertEqual(len(rows), 100)

    def test_sensevoice_coverage_alias_accepts_whole_rejects_excerpt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'report.json'
            source = str(Path(folder)/'canonical.wav')
            audio = {'file': source, 'sha256': 'fixture-sha', 'duration_seconds': 16}
            report = {'audio': {'source': source, 'sha256': 'fixture-sha',
                                'duration_seconds': 16, 'time_origin': 'canonical_wav_start'},
                      'coverage': {'selection_start_seconds': 0, 'selection_end_seconds': 16}}
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertEqual(load_bound_report(path, audio)['coverage']['selection_end_seconds'], 16)
            report['coverage']['selection_start_seconds'] = 8
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_bound_report(path, audio)


if __name__ == '__main__':
    unittest.main()
