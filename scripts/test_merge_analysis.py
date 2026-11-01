"""Self-contained contract tests using synthetic WAV and explicitly synthetic reports."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location("merge_analysis", Path(__file__).with_name("merge_analysis.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MergeContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bgm-merge-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, silent=False):
        audio = self.root / ("silence.wav" if silent else "step.wav")
        rate, duration = 8000, 8
        with wave.open(str(audio), "wb") as handle:
            handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            handle.writeframes(b"".join(struct.pack("<h", 0 if silent else round((.04 if i < 4*rate else .4)*32767*math.sin(2*math.pi*440*i/rate))) for i in range(rate*duration)))
        digest = module.sha256(audio)
        times = [] if silent else [round(.25+i*.25, 4) for i in range(31)]
        curve = [.142]*16 if silent else [.1]*8+[.6]*8
        reports = {
            "sonara": {"schemaVersion": 1, "audio": {"source": str(audio), "sha256": digest, "durationSeconds": 8, "timeOrigin": "input file start"},
                       "rhythm": {"bpmCandidate": 120, "beatsSeconds": [] if silent else list(range(8)), "downbeatsSeconds": []},
                       "dynamics": {"energyCurve": curve, "energyCurveHopSeconds": .5},
                       "segments": [{"start_sec": 0, "end_sec": 8, "energy": .142 if silent else .35}]},
            "beats": {"schemaVersion": 1, "sourceAudio": str(audio), "audioSha256": digest, "durationSeconds": 8,
                      "sampleRate": rate, "timeOrigin": "input WAV start", "model": "SYNTHETIC TEST FIXTURE",
                      "beatSeconds": [i*.5 for i in range(16)], "downbeatSeconds": [0, 2, 4, 6]},
            "onsets": {"schemaVersion": 1, "sourceAudio": str(audio), "durationSeconds": 8, "analysisOffset": 0,
                       "analysisDurationSeconds": 8, "analysisRangeSeconds": [0, 8],
                       "timeBase": "Seconds from the beginning of sourceAudio; analysisOffset already included in every event.",
                       "eventTypeAssumption": "onset", "events": [{"id": str(i), "timeSeconds": value, "strength": .5,
                            "confidence": "high", "accepted": True, "type": "onset"} for i, value in enumerate(times)]}}
        paths = {}
        for name, report in reports.items():
            paths[name] = self.root / (name+".json")
            paths[name].write_text(json.dumps(report), encoding="utf8")
        return reports, paths

    def test_short_energy_change_and_bounded_events_without_grid(self):
        reports, paths = self.fixture()
        output = module.merge(**reports, paths=paths, max_onsets=4)
        self.assertEqual(output["rhythm"]["beatSeconds"], reports["beats"]["beatSeconds"])
        self.assertEqual(len(output["onsets"]["strongestAccepted"]), 4)
        self.assertEqual(output["onsets"]["windowAcceptedCount"], 31)
        self.assertTrue(any(abs(item["timeSeconds"]-4) <= .5 for item in output["structure"]["shortWindowChangeCandidates"]))
        self.assertEqual(output["rhythm"]["tempo"]["relation"], "approximately_agree")
        self.assertNotIn("raw", output)

    def test_silence_suppresses_sonara_defaults(self):
        reports, paths = self.fixture(silent=True)
        output = module.merge(**reports, paths=paths)
        self.assertTrue(output["dynamics"]["waveformLevel"]["digitalSilence"])
        self.assertIsNone(output["rhythm"]["tempo"]["sonaraBpmCandidate"])
        self.assertIsNone(output["rhythm"]["tempo"]["beatThisMedianIntervalBpm"])
        self.assertIsNone(output["dynamics"]["energyStatistics"])
        self.assertEqual(output["structure"]["shortWindowChangeCandidates"], [])
        self.assertEqual(output["structure"]["longSectionCandidates"], [])
        self.assertEqual(output["onsets"]["strongestAccepted"], [])

    def test_wrong_hash_refused(self):
        reports, paths = self.fixture()
        reports["beats"]["audioSha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "hashes"):
            module.merge(**reports, paths=paths)

    def test_wrong_source_refused(self):
        reports, paths = self.fixture()
        reports["onsets"]["sourceAudio"] = str(paths["onsets"])
        with self.assertRaisesRegex(ValueError, "paths differ"):
            module.merge(**reports, paths=paths)

    def test_range_and_timestamp_refused(self):
        reports, paths = self.fixture()
        with self.assertRaisesRegex(ValueError, "covered"):
            module.merge(**reports, paths=paths, window_start=7, window_duration=2)
        reports["onsets"]["events"][0]["timeSeconds"] = 50
        with self.assertRaisesRegex(ValueError, "outside"):
            module.merge(**reports, paths=paths)

    def test_bpm_double_and_unsupported(self):
        reports, paths = self.fixture()
        reports["sonara"]["rhythm"]["bpmCandidate"] = 240
        self.assertEqual(module.merge(**reports, paths=paths)["rhythm"]["tempo"]["relation"], "sonara_approximately_double")
        reports["sonara"]["rhythm"]["beatsSeconds"] = []
        output = module.merge(**reports, paths=paths)
        self.assertIsNone(output["rhythm"]["tempo"]["sonaraBpmCandidate"])
        self.assertEqual(output["rhythm"]["tempo"]["relation"], "sonara_unknown_no_beat_support")

    def test_sonara_small_tail_logged_but_excessive_tail_refused(self):
        reports, paths = self.fixture()
        reports["sonara"]["engine"] = {"provenance": {"hop_length": 512, "sample_rate": 22050}}
        reports["sonara"]["rhythm"]["beatsSeconds"].append(8.035)
        output = module.merge(**reports, paths=paths)
        artifacts = output["verification"]["sonaraTailArtifacts"]
        self.assertEqual(artifacts[0]["removedSeconds"], [8.035])
        self.assertEqual(artifacts[0]["removedCount"], 1)
        reports["sonara"]["rhythm"]["beatsSeconds"][-1] = 8.2
        with self.assertRaisesRegex(ValueError, "outside"):
            module.merge(**reports, paths=paths)
        reports["sonara"]["rhythm"]["beatsSeconds"] = list(range(8))+[8.01, 8.02, 8.03]
        with self.assertRaisesRegex(ValueError, "excessive"):
            module.merge(**reports, paths=paths)

    def test_nonzero_onset_range_is_not_offset_twice(self):
        reports, paths = self.fixture()
        reports["onsets"].update(analysisOffset=2, analysisDurationSeconds=6, analysisRangeSeconds=[2, 8])
        reports["onsets"]["events"] = [event for event in reports["onsets"]["events"] if event["timeSeconds"] >= 2]
        output = module.merge(**reports, paths=paths, window_start=4, window_duration=2)
        self.assertEqual(output["audio"]["analysisWindowSeconds"], [4, 6])
        self.assertEqual(output["onsets"]["windowAcceptedCount"], 8)
        self.assertEqual(output["onsets"]["strongestAccepted"][0]["timeSeconds"], 4)
        self.assertEqual(output["rhythm"]["beatSeconds"], reports["beats"]["beatSeconds"])
        with self.assertRaisesRegex(ValueError, "finite"):
            module.merge(**reports, paths=paths, window_start=float("nan"))

    def test_cli_preserves_existing_output(self):
        reports, paths = self.fixture()
        output = self.root / "out.json"
        output.write_text("keep", encoding="utf8")
        command = [sys.executable, str(Path(__file__).with_name("merge_analysis.py"))]
        for name, path in paths.items():
            command += ["--"+name, str(path)]
        result = subprocess.run(command+["--output", str(output)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overwrite", result.stderr)
        self.assertEqual(output.read_text(encoding="utf8"), "keep")


if __name__ == "__main__":
    unittest.main()
