"""Small numerical and time-contract checks; full audio trials remain separate."""
import unittest
import numpy as np
import analyze_semantics as sem


class SemanticFeaturesTest(unittest.TestCase):
    def test_official_patch_padding_does_not_add_nearly_empty_tail(self):
        self.assertEqual(sem.yamnet_patch_starts(16000), [0, 7680])
        self.assertEqual(sem.yamnet_patch_starts(15600), [0])
        self.assertEqual(sem.yamnet_patch_starts(100), [0])

    def test_yamnet_shape_and_silence_log_floor(self):
        result = sem.yamnet_patches(np.zeros(16000, np.float32), [0, 7680])
        self.assertEqual(result.shape, (2, 1, 96, 64))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result, np.log(.001), atol=1e-6)

    def test_mel_sine_localizes_frequency(self):
        t = np.arange(16000) / 16000
        result = sem.musicnn_spectrogram((.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
        self.assertEqual(result.shape[1], 96)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result >= 0).all())
        # Low-frequency sine should concentrate below the middle of the mel range.
        self.assertLess(int(result.mean(axis=0).argmax()), 40)
        self.assertGreater(float(result.max()), 2)

    def test_voice_segments_preserve_gaps_and_type_changes(self):
        def row(start, end, activity, score):
            return dict(start=start, end=end, activity=activity, vocal_score=score)
        result = sem.group_activity([row(0, .48, 'speech_candidate', .7),
                                     row(.48, .96, 'speech_candidate', .9),
                                     row(.96, 1.44, 'uncertain', .2),
                                     row(1.44, 1.92, 'singing_candidate', .8)])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {'start': 0, 'end': .96, 'type': 'speech_candidate', 'peak_score': .9})
        self.assertEqual(result[1]['start'], 1.44)

    def test_uncertain_vocal_evidence_is_retained_as_uncertain(self):
        result = sem.group_activity([dict(start=0, end=.48, activity='uncertain_vocal_candidate', vocal_score=.95)])
        self.assertEqual(result[0]['type'], 'uncertain_vocal_candidate')


if __name__ == '__main__':
    unittest.main()
