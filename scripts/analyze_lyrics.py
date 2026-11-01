"""Offline CPU Whisper transcription with source-bound, provisional phrase timings.

Run after semantic classification when available: a failed or unsupported transcript
is represented explicitly rather than filling instrumental passages with words.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audio_identity(path):
    with wave.open(str(path), "rb") as source:
        duration = source.getnframes() / source.getframerate()
    return {"source": str(path.resolve()), "sha256": digest(path),
            "duration_seconds": duration, "time_origin": "canonical_wav_start"}


def vocal_evidence(timeline, start, end):
    weight = 0.0
    score = speech = singing = 0.0
    peak = 0.0
    for row in timeline:
        overlap = max(0.0, min(end, float(row["end"])) - max(start, float(row["start"])))
        if overlap <= 0:
            continue
        value = float(row.get("vocal_score", 0))
        weight += overlap
        score += overlap * value
        speech += overlap * float(row.get("speech_score", 0))
        singing += overlap * float(row.get("singing_score", 0))
        peak = max(peak, value)
    if not weight:
        return None
    return {"vocal_score_mean": round(score / weight, 4),
            "vocal_score_peak": round(peak, 4),
            "speech_score_mean": round(speech / weight, 4),
            "singing_score_mean": round(singing / weight, 4)}


def repetition_ratio(text):
    text = re.sub(r"\s+", "", text)
    if len(text) < 12:
        return 0.0
    # Repeated token/trigram loops are a common decoder failure on accompaniment.
    grams = [text[i:i + 3] for i in range(len(text) - 2)]
    return 1 - len(set(grams)) / len(grams)


def select_vocal_intervals(timeline, start, end):
    """Use independent vocal evidence for sparse ASR, retaining source offsets."""
    if not timeline:
        return [(start, end)]
    intervals = []
    for row in timeline:
        if float(row.get("vocal_score", 0)) < .10:
            continue
        left, right = max(start, float(row["start"]) - .45), min(end, float(row["end"]) + .75)
        if right <= left:
            continue
        if intervals and left - intervals[-1][1] <= 1.0:
            intervals[-1] = (intervals[-1][0], max(right, intervals[-1][1]))
        else:
            intervals.append((left, right))
    return intervals


def transcription_jobs(intervals, maximum_seconds=20.0, context_seconds=.5):
    """Bound per-call memory, retaining each word in one nonoverlapping range."""
    jobs = []
    for left, right in intervals:
        cursor = left
        while cursor < right - 1e-6:
            core_end = min(right, cursor + maximum_seconds)
            jobs.append({"start": max(left, cursor - context_seconds),
                         "end": min(right, core_end + context_seconds),
                         "retain_start": cursor, "retain_end": core_end})
            cursor = core_end
    return jobs


def evaluate_segment(segment, timeline):
    evidence = vocal_evidence(timeline, segment["start"], segment["end"])
    reasons = []
    logprob = float(segment.get("avg_logprob", -99))
    compression = float(segment.get("compression_ratio", 0))
    no_speech = float(segment.get("no_speech_prob", 0))
    if logprob < -1.0:
        reasons.append("low_decoder_logprob")
    if compression > 2.4 or repetition_ratio(segment.get("text", "")) > .65:
        reasons.append("repetitive_decoder_output")
    if no_speech > .6 and logprob < -.5:
        reasons.append("decoder_no_speech_with_weak_text")
    if evidence and evidence["vocal_score_peak"] < .10:
        reasons.append("no_supporting_vocal_classifier_evidence")
    return reasons, evidence


def phrase_segments(raw_segments, timeline, duration):
    accepted, rejected = [], []
    for raw in raw_segments:
        start, end = max(0.0, float(raw["start"])), min(duration, float(raw["end"]))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        segment = {**raw, "start": start, "end": end}
        reasons, evidence = evaluate_segment(segment, timeline)
        if reasons:
            rejected.append({"start": round(start, 3), "end": round(end, 3),
                             "text": raw.get("text", "").strip(), "reasons": reasons,
                             "vocal_evidence": evidence})
            continue
        words = []
        for word in raw.get("words", []):
            ws, we = max(start, float(word["start"])), min(end, float(word["end"]))
            if math.isfinite(ws) and math.isfinite(we) and we > ws:
                words.append({"text": word.get("word", ""), "start": round(ws, 3),
                              "end": round(we, 3),
                              "score": round(float(word.get("probability", 0)), 4)})
        groups, current = [], []
        for word in words:
            if current and (word["start"] - current[-1]["end"] > .55
                            or word["end"] - current[0]["start"] > 12):
                groups.append(current)
                current = []
            current.append(word)
            if re.search(r"[,，。.!！?？;；:：]$", word["text"].strip()):
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        if not groups:
            groups = [[{"text": raw.get("text", ""), "start": round(start, 3),
                        "end": round(end, 3), "score": None}]]
        for group in groups:
            text = "".join(w["text"] for w in group).strip()
            if not text:
                continue
            ps, pe = group[0]["start"], group[-1]["end"]
            ev = vocal_evidence(timeline, ps, pe)
            if ev and (ev["vocal_score_peak"] < .10
                       or (ev["vocal_score_mean"] < .03 and ev["vocal_score_peak"] < .30)):
                rejected.append({"start": ps, "end": pe, "text": text,
                                 "reasons": ["phrase_without_supporting_vocal_evidence"],
                                 "vocal_evidence": ev})
                continue
            kind = "vocal_text_candidate"
            if ev:
                sp, si = ev["speech_score_mean"], ev["singing_score_mean"]
                if max(sp, si) >= .15:
                    kind = "speech_candidate" if sp > si * 1.4 else (
                        "singing_candidate" if si > sp * 1.4 else "mixed_vocal_candidate")
            accepted.append({"start": ps, "end": pe, "text": text, "type": kind,
                             "words": group, "needs_review": True,
                             "vocal_evidence": ev,
                             "quality": {"decoder_avg_logprob": round(float(raw.get("avg_logprob", 0)), 4),
                                         "decoder_no_speech_score": round(float(raw.get("no_speech_prob", 0)), 4),
                                         "compression_ratio": round(float(raw.get("compression_ratio", 0)), 4),
                                         "status": "unverified_asr_candidate"}})
    return accepted, rejected


def refine_phrase_fragments(segments):
    """Join adjacent recognized function-word lead-ins; quarantine orphan ones."""
    prefixes = {"i", "i'm", "i’m", "i've", "i’ve", "i'll", "i’ll", "i'd", "i’d",
                "you", "you're", "you’re", "we", "we're", "we’re", "he", "she", "they",
                "the", "a", "an", "and", "to", "of", "it's", "it’s"}
    refined, fragments = [], []
    index = 0
    while index < len(segments):
        current = dict(segments[index])
        scores = [w["score"] for w in current.get("words", [])
                  if isinstance(w.get("score"), (int, float)) and math.isfinite(w["score"])]
        current["quality"] = dict(current.get("quality", {}))
        if scores:
            current["quality"]["word_score_mean"] = round(sum(scores) / len(scores), 4)
            current["quality"]["word_score_min"] = round(min(scores), 4)
        normalized = current["text"].strip().strip(".,，。!！?？;；:").lower()
        if normalized in prefixes:
            if index + 1 < len(segments):
                following = segments[index + 1]
                gap = following["start"] - current["end"]
                if -.05 <= gap <= .8 and following["end"] - current["start"] <= 12:
                    current["joined_from"] = [{"start": current["start"], "end": current["end"], "text": current["text"]},
                                              {"start": following["start"], "end": following["end"], "text": following["text"]}]
                    current["end"] = following["end"]
                    current["text"] = current["text"].rstrip() + " " + following["text"].lstrip()
                    current["words"] = current.get("words", []) + following.get("words", [])
                    merged_scores = [w["score"] for w in current["words"]
                                     if isinstance(w.get("score"), (int, float)) and math.isfinite(w["score"])]
                    if merged_scores:
                        current["quality"]["word_score_mean"] = round(sum(merged_scores) / len(merged_scores), 4)
                        current["quality"]["word_score_min"] = round(min(merged_scores), 4)
                    current["quality"]["phrase_grouping"] = "Adjacent recognized function-word lead-in joined; no words inserted"
                    current["needs_review"] = True
                    refined.append(current)
                    index += 2
                    continue
            current["fragment"] = True
            current["fragment_reason"] = "Isolated function word without nearby recognized completion"
            current["use_as_standalone_lyric"] = False
            fragments.append(current)
        else:
            refined.append(current)
        index += 1
    return refined, fragments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, default=Path("D:/SelfMedia/管理/工具/bgm-analysis"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--model", choices=["tiny", "base", "small"], default="base")
    p.add_argument("--model-root", type=Path)
    p.add_argument("--semantic-file", type=Path)
    p.add_argument("--vocal-wav", type=Path)
    p.add_argument("--language")
    p.add_argument("--language-evidence", type=Path, action="append", default=[], help="Prior same-audio transcript JSON whose language candidate should be checked/reused.")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float)
    p.add_argument("--initial-prompt")
    p.add_argument("--resume-raw", type=Path, help="Resume an identity-checked partial raw checkpoint.")
    p.add_argument("--beam-size", type=int, help="Default 1 for small, 5 for tiny/base.")
    p.add_argument("--max-tokens", type=int, help="Default 128 for small, 224 for tiny/base.")
    p.add_argument("--tokens-per-second", type=float, help="Default 6 for small, 12 for tiny/base.")
    p.add_argument("--allow-decode-change", action="store_true", help="Explicitly allow only beam/token-budget changes while resuming.")
    a = p.parse_args()
    if a.beam_size is None:
        a.beam_size = 1 if a.model == "small" else 5
    if a.max_tokens is None:
        a.max_tokens = 128 if a.model == "small" else 224
    if a.tokens_per_second is None:
        a.tokens_per_second = 6 if a.model == "small" else 12
    identity = audio_identity(a.wav)
    duration = identity["duration_seconds"]
    if (not math.isfinite(a.start) or a.start < 0 or a.start >= duration
            or (a.duration is not None and (not math.isfinite(a.duration) or a.duration <= 0))):
        p.error("Invalid analysis range.")
    if not 1 <= a.threads <= 8:
        p.error("threads must be between 1 and 8.")
    if not 1 <= a.beam_size <= 5 or not 24 <= a.max_tokens <= 224 or not 1 <= a.tokens_per_second <= 20:
        p.error("Invalid decoder resource budget.")
    end = min(duration, a.start + a.duration) if a.duration else duration
    timeline = []
    if a.semantic_file:
        semantics = read_json(a.semantic_file)
        if (semantics["audio"]["sha256"] != identity["sha256"]
                or abs(float(semantics["audio"]["duration_seconds"]) - duration) > .05):
            raise ValueError("Semantic analysis belongs to a different canonical WAV.")
        timeline = semantics.get("vocals", {}).get("timeline", [])
    language_evidence = []
    for evidence_path in a.language_evidence:
        evidence = read_json(evidence_path)
        if evidence.get("audio", {}).get("sha256") != identity["sha256"]:
            raise ValueError("Language evidence belongs to a different canonical WAV.")
        language_evidence.append({"file": str(evidence_path.resolve()), "sha256": digest(evidence_path),
                                  "language": evidence.get("language"), "coverage": evidence.get("coverage")})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = a.output.with_name(a.output.stem + "-raw.json")
    model_root = a.model_root
    if model_root is None:
        candidates = [a.runtime_root / "lyrics/models", Path("D:/C-drive-relocated/13779/whisper")]
        model_root = next((root for root in candidates if (root / (a.model + ".pt")).is_file()), candidates[0])
    model_path = model_root / (a.model + ".pt")
    if not model_path.is_file():
        raise FileNotFoundError(f"Offline ASR model missing: {model_path}")
    result = {"schema": "bgm-vocal-transcript-v1", "audio": identity,
              "status": "complete", "language": None,
              "coverage": {"start": a.start, "end": end},
              "coverage_meaning": "Processing range; retained text candidates are not guaranteed to cover every sung/spoken phrase.",
              "model": {"name": "OpenAI Whisper " + a.model, "file": str(model_path),
                        "sha256": digest(model_path), "device": "cpu", "threads": a.threads,
                        "source": "https://github.com/openai/whisper",
                        "initial_prompt": a.initial_prompt, "language_requested": a.language,
                        "temperature": 0, "beam_size": a.beam_size,
                        "max_tokens": a.max_tokens, "tokens_per_second": a.tokens_per_second,
                        "condition_on_previous_text": False},
              "segments": [], "rejected_segments": [], "human_reviewed": False,
              "timing_meaning": "ASR word/phrase alignment candidates, not measured syllable boundaries.",
              "text_meaning": "Automatically transcribed words; homophones, singing and mixing can cause errors. Scores are decoder diagnostics, not calibrated correctness probabilities.",
              "raw_file": str(raw_path.resolve())}
    result["language_evidence"] = language_evidence
    if a.semantic_file:
        result["semantic_source"] = {"file": str(a.semantic_file.resolve()),
                                     "sha256": digest(a.semantic_file),
                                     "meaning": "Independent classifier candidates used for ASR routing and rejection; not vocal-presence ground truth."}
    scoped_timeline = [r for r in timeline if r["end"] > a.start and r["start"] < end]
    if scoped_timeline and max(float(r.get("vocal_score", 0)) for r in scoped_timeline) < .10:
        result["status"] = "no_supported_transcript"
        result["reason"] = "No supporting vocal evidence in independent sound classifier; ASR skipped to avoid instrumental hallucinations. This is not proof of absence."
        raw_path.write_text(json.dumps({"skipped": True, "reason": result["reason"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(a.output), "status": result["status"], "segments": 0}))
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(a.threads)
    import imageio_ffmpeg
    import numpy as np
    import torch
    import whisper
    torch.set_num_threads(a.threads)
    source_path = a.vocal_wav or a.wav
    source_reason = "Provided separated vocal track" if a.vocal_wav else "Canonical audio mix"
    if a.vocal_wav and scoped_timeline:
        mean_speech = sum(float(r.get("speech_score", 0)) for r in scoped_timeline) / len(scoped_timeline)
        mean_singing = sum(float(r.get("singing_score", 0)) for r in scoped_timeline) / len(scoped_timeline)
        if mean_speech > .5 and mean_speech > mean_singing * 2:
            source_path = a.wav
            source_reason = "Strong predominantly spoken evidence; retain canonical speech to avoid separator artifacts"
    intervals = select_vocal_intervals(scoped_timeline, a.start, end)
    result["transcription_windows"] = [{"start": round(left, 3), "end": round(right, 3)}
                                       for left, right in intervals]
    result["asr_chunking"] = {"maximum_core_seconds": 20, "context_seconds": .5,
                              "boundary_handling": "Word midpoints retained in nonoverlapping core ranges; boundary wording remains provisional."}
    result["coverage_meaning"] = ("Canonical range classified for vocals; ASR runs only the listed independent vocal-candidate windows with context. "
                                  "Low-score or wordless vocals can remain untranscribed; text is not guaranteed exhaustive.") if scoped_timeline else result["coverage_meaning"]

    def decode_range(left, right):
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", str(left),
                   "-i", str(source_path), "-t", str(right - left), "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"]
        waveform = np.frombuffer(subprocess.run(command, check=True, capture_output=True).stdout, dtype=np.float32).copy()
        if abs(len(waveform) / 16000 - (right - left)) > .1:
            raise ValueError("Transcription audio duration differs from the canonical analysis range.")
        return waveform
    started = time.monotonic()
    model = whisper.load_model(str(model_path), device="cpu")
    # Loading an explicit offline path does not attach Whisper's calibrated
    # alignment-head selection; restore the official mask for this model size.
    alignment_heads = getattr(whisper, "_ALIGNMENT_HEADS", {}).get(a.model)
    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)
    jobs = transcription_jobs(intervals)
    inputs = {"audio_sha256": identity["sha256"], "model_sha256": result["model"]["sha256"],
              "semantic_sha256": result.get("semantic_source", {}).get("sha256"),
              "transcription_audio_sha256": digest(source_path), "range": [a.start, end],
              "initial_prompt": a.initial_prompt, "language_requested": a.language,
              "language_evidence_sha256": [row["sha256"] for row in language_evidence] or None,
              "jobs": jobs, "beam_size": a.beam_size,
              "max_tokens": a.max_tokens, "tokens_per_second": a.tokens_per_second,
              "temperature": 0, "condition_on_previous_text": False}
    resume = read_json(a.resume_raw) if a.resume_raw else None
    decode_change = None
    if resume:
        prior_inputs = dict(resume.get("inputs", {}))
        # These constants were fixed before the corresponding CLI options existed.
        prior_inputs.setdefault("max_tokens", 224)
        prior_inputs.setdefault("tokens_per_second", 12)
        changed = {key for key in set(prior_inputs) | set(inputs) if prior_inputs.get(key) != inputs.get(key)}
        allowed = {"beam_size", "max_tokens", "tokens_per_second"}
        if changed and not (a.allow_decode_change and changed <= allowed):
            raise ValueError("Partial ASR checkpoint identity mismatch in: " + ", ".join(sorted(changed)))
        if changed:
            decode_change = {"start_window": len(resume.get("transcription_windows", [])) + 1,
                             "previous": {key: prior_inputs[key] for key in allowed},
                             "current": {key: inputs[key] for key in allowed},
                             "explicitly_allowed": True, "reason": "Explicit decoder performance change; audio/model/semantic/range/window identities unchanged"}
        completed = resume.get("transcription_windows", [])
        if len(completed) > len(jobs) or any(
                any(abs(float(row[key]) - float(job[key])) > 1e-6 for key in ("start", "end", "retain_start", "retain_end"))
                for row, job in zip(completed, jobs)):
            raise ValueError("Partial ASR checkpoint is not a prefix of this analysis job list.")
        result["resumed_from"] = {"file": str(a.resume_raw.resolve()), "sha256": digest(a.resume_raw),
                                   "completed_asr_windows": len(completed)}
    evidence_languages = {row["language"] for row in language_evidence if row["language"]}
    evidence_language = next(iter(evidence_languages)) if len(evidence_languages) == 1 else None
    language = a.language or evidence_language or (resume.get("language") if resume else None)
    if language_evidence:
        result["language_detection"] = {"method": "Explicit language checked against prior same-audio ASR evidence" if a.language else "Reused unanimous same-audio ASR language candidate",
                                        "language": language, "evidence_agrees": evidence_languages == {language}}
    if resume and language:
        result["language_detection"] = resume.get("language_detection", {
            "method": "Language reused from identity-checked partial Whisper run", "language": language,
            "scores_unavailable": True})
    if not language and scoped_timeline and model.is_multilingual:
        strongest = max(scoped_timeline, key=lambda r: float(r.get("vocal_score", 0)))
        center = (strongest["start"] + strongest["end"]) / 2
        selected = next(((left, right) for left, right in intervals if left <= center <= right), intervals[0])
        detect_start = max(selected[0], min(selected[1] - 30, center - 15))
        detect_end = min(selected[1], detect_start + 30)
        language_windows = [(detect_start, detect_end)]
        first = next(((left, right) for left, right in intervals if right - left >= 8), intervals[0])
        first_window = (first[0], min(first[1], first[0] + 30))
        overlap = max(0, min(detect_end, first_window[1]) - max(detect_start, first_window[0]))
        if overlap < min(detect_end - detect_start, first_window[1] - first_window[0]) * .5:
            language_windows.insert(0, first_window)
        votes = []
        window_reports = []
        for left, right in language_windows:
            clip = whisper.pad_or_trim(decode_range(left, right))
            mel = whisper.log_mel_spectrogram(clip, n_mels=model.dims.n_mels).to(model.device)
            _, probabilities = model.detect_language(mel)
            votes.append(probabilities)
            window_reports.append({"start": left, "end": right,
                                   "top_candidates": [{"language": lang, "score": round(score, 4)}
                                                      for lang, score in sorted(probabilities.items(), key=lambda x: -x[1])[:3]]})
        probabilities = {key: sum(vote[key] for vote in votes) / len(votes) for key in votes[0]}
        language = max(probabilities, key=probabilities.get)
        result["language_detection"] = {
            "window_start": round(detect_start, 3), "window_end": round(detect_end, 3),
            "method": "Mean of distinct sustained/strong vocal-context language candidates; not a confirmed language",
            "windows": window_reports,
            "top_candidates": [{"language": lang, "score": round(score, 4)}
                               for lang, score in sorted(probabilities.items(), key=lambda x: -x[1])[:3]]}
    raw = resume or {"text": "", "segments": [], "language": language, "transcription_windows": [],
                     "processing_complete": False, "expected_asr_windows": len(jobs), "inputs": inputs,
                     "language_detection": result.get("language_detection")}
    completed_count = len(raw["transcription_windows"])
    if resume:
        for row in raw["transcription_windows"]:
            row.setdefault("decode_config", {"beam_size": prior_inputs["beam_size"], "sample_len": row["sample_len"]})
        raw["inputs"] = inputs
    if decode_change:
        raw.setdefault("decode_history", []).append(decode_change)
    raw["processing_complete"] = False
    for index, job in enumerate(jobs, 1):
        if index <= completed_count:
            continue
        left, right = job["start"], job["end"]
        print(f"ASR window {index}/{len(jobs)}: {left:.3f}-{right:.3f}s", file=sys.stderr, flush=True)
        waveform = decode_range(left, right)
        token_limit = min(a.max_tokens, max(24, math.ceil((right - left) * a.tokens_per_second)))
        part = model.transcribe(waveform, language=language, fp16=False,
                                word_timestamps=True, verbose=None, temperature=0,
                                beam_size=a.beam_size, condition_on_previous_text=False,
                                initial_prompt=a.initial_prompt, hallucination_silence_threshold=1.0,
                                sample_len=token_limit)
        language = language or part.get("language")
        raw["language"] = language
        raw["text"] += part.get("text", "")
        raw["transcription_windows"].append({**job, "sample_len": token_limit,
                                             "decode_config": {"beam_size": a.beam_size, "sample_len": token_limit}})
        for seg in part["segments"]:
            seg["start"] = max(job["retain_start"], seg["start"] + left)
            seg["end"] = min(job["retain_end"], seg["end"] + left)
            seg["analysis_window"] = {"start": left, "end": right}
            seg["retained_window"] = {"start": job["retain_start"], "end": job["retain_end"]}
            all_words = seg.get("words", [])
            for word in seg.get("words", []):
                word["start"] += left
                word["end"] += left
            seg["words"] = [word for word in all_words
                            if job["retain_start"] <= (word["start"] + word["end"]) / 2 < job["retain_end"]]
            if all_words and not seg["words"]:
                continue
            raw["segments"].append(seg)
        temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
        temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(raw_path)
    raw["processing_complete"] = True
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    result["language"] = raw.get("language")
    result["decode_history"] = raw.get("decode_history", [])
    result["decode_configuration_scope"] = "Current model decoder settings apply to new windows; resumed earlier windows retain explicit per-window decode_config in raw_file."
    result["segments"], result["rejected_segments"] = phrase_segments(raw["segments"], timeline, duration)
    result["segments"], result["fragments"] = refine_phrase_fragments(result["segments"])
    result["fragment_policy"] = "Adjacent recognized lead-ins may be grouped; orphan function-word fragments remain outside standalone lyric candidates. No missing words are invented."
    result["status"] = "complete" if result["segments"] else "no_supported_transcript"
    if not result["segments"]:
        result["reason"] = "ASR produced no supported words. Vocals may be wordless, masked, in an uncertain language, or missed; consult the independent vocal timeline."
    result["analysis_elapsed_seconds"] = round(time.monotonic() - started, 2)
    result["transcription_audio"] = {"file": str(source_path.resolve()),
                                      "is_separated_vocal_track": source_path != a.wav,
                                      "selection_reason": source_reason,
                                      "sha256": digest(source_path)}
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(a.output), "status": result["status"],
                      "language": result["language"], "segments": len(result["segments"]),
                      "rejected_segments": len(result["rejected_segments"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
