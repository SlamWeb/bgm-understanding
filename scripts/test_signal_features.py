"""Small signal-level invariants, not music-perception accuracy tests."""
import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from analyze_signal_features import analyze


class SignalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bgm-features-")
        self.root = Path(self.temp.name)
        self.rate = 44100

    def tearDown(self):
        self.temp.cleanup()

    def wav(self, name, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        path = self.root / name
        with wave.open(str(path), "wb") as w:
            w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(self.rate)
            w.writeframes(np.clip(np.rint(x * 32768), -32768, 32767).astype("<i2").tobytes())
        return path

    def test_silence_has_null_dbfs_and_no_changes(self):
        d = analyze(self.wav("zero.wav", np.zeros(self.rate * 3)))
        self.assertEqual(d["changes"], [])
        self.assertTrue(all(w["rms"] == 0 and w["dbfs"] is None for w in d["windows"]))

    def test_antiphase_does_not_cancel(self):
        t = np.arange(self.rate * 2) / self.rate
        x = 0.2 * np.sin(2 * np.pi * 1000 * t)
        mono = analyze(self.wav("mono.wav", x))
        stereo = analyze(self.wav("opposed.wav", np.stack([x, -x], axis=1)))
        self.assertGreater(stereo["windows"][1]["rms"], 0.13)
        self.assertAlmostEqual(mono["windows"][1]["rms"], stereo["windows"][1]["rms"], places=5)
        self.assertAlmostEqual(mono["windows"][1]["band_rms"]["mid"], stereo["windows"][1]["band_rms"]["mid"], places=5)

    def test_fixed_level_frequency_switch_is_found_at_source_time(self):
        t = np.arange(self.rate * 4) / self.rate
        x = np.where(t < 2, 0.2 * np.sin(2 * np.pi * 100 * t), 0.2 * np.sin(2 * np.pi * 4000 * t))
        d = analyze(self.wav("switch.wav", x))
        found = [c for c in d["changes"] if abs(c["seconds"] - 2.0) < 0.01]
        self.assertEqual(len(found), 1)
        self.assertIn("frequency_change", found[0]["types"])
        self.assertNotIn("energy_increase", found[0]["types"])
        self.assertNotIn("energy_decrease", found[0]["types"])
        self.assertLess(found[0]["frequency_share_delta"]["low"], -0.7)
        self.assertGreater(found[0]["frequency_share_delta"]["high"], 0.7)
        self.assertEqual(found[0]["before_after"]["before"]["range_seconds"], [1.0, 2.0])
        self.assertEqual(found[0]["before_after"]["after"]["range_seconds"], [2.0, 3.0])

    def test_tail_and_very_short_input_are_covered(self):
        count = self.rate + 13
        d = analyze(self.wav("tail.wav", np.zeros(count)))
        self.assertEqual(sum(w["valid_samples"] for w in d["windows"]), count)
        self.assertAlmostEqual(d["windows"][-1]["end"], count / self.rate, places=6)
        short = analyze(self.wav("tiny.wav", np.zeros(7)))
        self.assertEqual(len(short["windows"]), 1)
        self.assertEqual(short["windows"][0]["valid_samples"], 7)
        self.assertEqual(short["changes"], [])

    def test_cli_refuses_overwrite(self):
        path = self.wav("input.wav", np.zeros(self.rate))
        target = self.root / "output.json"
        target.write_text("preserve-me", encoding="utf-8")
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("analyze_signal_features.py")),
                                 "--wav", str(path), "--output", str(target)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_text(encoding="utf-8"), "preserve-me")


if __name__ == "__main__":
    unittest.main()
