"""Timing/coverage invariants for speech semantics, without loading a model."""
import unittest
import numpy as np

from analyze_speech_semantics import speech_chunks, split_tokens


class SpeechSemanticsTests(unittest.TestCase):
    def test_silence_does_not_create_transcription_chunks(self):
        self.assertEqual(speech_chunks(np.zeros(32000, np.float32)), [])

    def test_energy_pause_bounds_chunks_and_retains_speech(self):
        samples = np.zeros(16000 * 4, np.float32)
        samples[3200:16000] = .2
        samples[32000:48000] = .2
        rows = speech_chunks(samples)
        self.assertEqual(len(rows), 2)
        self.assertLessEqual(rows[0][0], 3200)
        self.assertGreaterEqual(rows[0][1], 16000)
        self.assertLess(rows[0][1], rows[1][0])
        self.assertLessEqual(rows[1][0], 32000)
        self.assertGreaterEqual(rows[1][1], 48000)

    def test_long_speech_chunks_have_bounded_context(self):
        rows = speech_chunks(np.ones(16000 * 45, np.float32), max_seconds=20)
        self.assertEqual(rows[0][0], 0)
        self.assertEqual(rows[-1][1], 16000 * 45)
        self.assertTrue(all(end - start <= 16000 * 20 for start, end in rows))

    def test_token_offsets_and_estimated_ends_are_preserved(self):
        rows = split_tokens(['春', '风', '。'], [.1, .3, .6], 10, 11)
        self.assertEqual(rows[0]['start'], 10.1)
        self.assertEqual(rows[0]['text'], '春风。')
        self.assertLessEqual(rows[0]['end'], 11)
        self.assertIn('estimated', rows[0]['words'][0]['end_method'])

    def test_nonfinite_or_different_length_timestamps_are_rejected(self):
        with self.assertRaises(ValueError):
            split_tokens(['a'], [float('nan')], 0, 1)
        with self.assertRaises(ValueError):
            split_tokens(['a', 'b'], [0.1], 0, 1)

    def test_reverse_token_timestamps_are_rejected(self):
        with self.assertRaises(ValueError):
            split_tokens(['a', 'b'], [.5, .1], 0, 1)


if __name__ == '__main__':
    unittest.main()
