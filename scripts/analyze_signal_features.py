"""Measure WAV level/frequency trajectories on CPU; output changes, not emotions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from pathlib import Path

import numpy as np
from scipy import signal


def read_pcm16(path: Path):
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Requires uncompressed PCM16 WAV; decode input first.")
        rate, channels, count = wav.getframerate(), wav.getnchannels(), wav.getnframes()
        if rate < 6000 or channels < 1 or not 0 < count / rate <= 600.001:
            raise ValueError("Requires nonempty audio <=600 seconds and sample rate >=6000 Hz.")
        raw = wav.readframes(count)
    if len(raw) != count * channels * 2:
        raise ValueError("Truncated WAV sample data.")
    x = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float64) / 32768.0
    return x, rate


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def dbfs(value):
    return 20.0 * math.log10(value) if value > 0 else None


def analyze(path: Path):
    path = Path(path).resolve()
    x, rate = read_pcm16(path)
    duration = len(x) / rate
    band_edges = {"low": [30.0, 180.0], "mid": [180.0, 2500.0], "high": [2500.0, min(10000.0, rate * 0.49)]}
    block = max(1, round(rate * 0.5))
    bins = []
    for i in range(0, len(x), block):
        end = min(len(x), i + block)
        value = rms(x[i:end])
        bins.append({"start": round(i / rate, 6), "end": round(end / rate, 6),
                     "rms": value, "dbfs": dbfs(value),
                     "band_rms": {},
                     "valid_samples": end - i})

    # Retain one filtered waveform at a time so a full song does not keep
    # four sample-sized arrays alive for all three bands.
    for name, (lo, hi) in band_edges.items():
        if hi <= lo:
            raise ValueError("Sample rate too low for requested frequency bands.")
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=rate, output="sos")
        filtered = signal.sosfilt(sos, x, axis=0)
        for idx, i in enumerate(range(0, len(x), block)):
            bins[idx]["band_rms"][name] = rms(filtered[i:min(len(x), i + block)])
        del filtered

    def aggregate(rows):
        weights = np.asarray([b["valid_samples"] for b in rows], dtype=np.float64)
        def weighted_rms(values):
            return float(np.sqrt(np.average(np.square(values), weights=weights)))
        value = weighted_rms([b["rms"] for b in rows])
        brms = {name: weighted_rms([b["band_rms"][name] for b in rows]) for name in band_edges}
        total_band_power = sum(v * v for v in brms.values())
        shares = {name: v * v / total_band_power if total_band_power > 0 else 0.0 for name, v in brms.items()}
        return {"range_seconds": [rows[0]["start"], rows[-1]["end"]], "rms": value,
                "dbfs": dbfs(value), "band_rms": brms, "band_power_share": shares}

    candidates = []
    floor, signal_floor = 1e-5, 1e-3
    # Consecutive ~1 second summaries around a 0.5 second boundary, retaining
    # exact range support when the last window is shorter.
    for i in range(2, len(bins) - 1):
        before, after = aggregate(bins[i-2:i]), aggregate(bins[i:i+2])
        if max(before["rms"], after["rms"]) < signal_floor:
            continue
        delta = 20 * math.log10(max(after["rms"], floor) / max(before["rms"], floor))
        share_delta = {name: after["band_power_share"][name] - before["band_power_share"][name] for name in band_edges}
        active_share_delta = max((abs(share_delta[name]) for name in band_edges
                                  if max(before["band_rms"][name], after["band_rms"][name]) >= signal_floor), default=0.0)
        types = []
        if abs(delta) >= 3.0:
            types.append("energy_increase" if delta > 0 else "energy_decrease")
        if active_share_delta >= 0.25:
            types.append("frequency_change")
        if not types:
            continue
        candidates.append({"seconds": bins[i]["start"], "types": types,
                           "before_after": {"before": before, "after": after},
                           "level_delta_db": delta, "frequency_share_delta": share_delta,
                           "score": max(abs(delta) / 3.0, active_share_delta / 0.25)})

    selected = []
    for c in sorted(candidates, key=lambda row: (-row["score"], row["seconds"])):
        if all(abs(c["seconds"] - old["seconds"]) >= 1.5 for old in selected):
            selected.append(c)
        if len(selected) >= 8:
            break
    selected.sort(key=lambda c: c["seconds"])
    return {"schema": "bgm-signal-features-v1",
            "audio": {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "duration_seconds": duration, "time_origin": "input WAV start", "sample_rate": rate,
                      "channels": int(x.shape[1])},
            "measurement": {"window_seconds": 0.5, "comparison_side_seconds": 1.0,
                            "band_edges_hz": band_edges, "channel_aggregation": "root mean square across samples and channels",
                            "change_delta_floor_dbfs": -100.0, "candidate_signal_floor_rms": signal_floor,
                            "minimum_level_delta_db": 3.0, "minimum_band_power_share_delta": 0.25,
                            "minimum_candidate_spacing_seconds": 1.5, "maximum_candidates": 8,
                            "candidate_score": "uncalibrated threshold exceedance, not perceptual importance"},
            "windows": bins, "changes": selected,
            "limitations": ["Windows measure signal level and frequency content, not instruments, emotions, chorus, or mandatory cuts.",
                            "Changes summarize about one second on each side and occur on a 0.5 second grid; they are not exact sound onsets.",
                            "Band RMS uses causal fourth-order Butterworth filters started at zero state; the beginning and filter transients can affect values.",
                            "Band power shares are normalized within the three filtered bands, not percentages of total signal energy.",
                            "Silent window dBFS is null. Change deltas use an explicit -100 dBFS floor when a side is silent.",
                            "Candidate thresholds and ranking are heuristic; a missing candidate does not establish absence of musical change."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file.")
    result = analyze(args.wav)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output.resolve()), "windows": len(result["windows"]),
                      "change_candidates": len(result["changes"])}))


if __name__ == "__main__":
    main()
