"""Offline, CPU-only All-In-One structure analysis for a canonical PCM WAV.

Reuses the installed D-drive third-party inference runtime and official weights.
Raw model output is preserved; downstream code must validate/summarize its times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import wave

DEFAULT_RUNTIME = Path("D:/SelfMedia/管理/工具/bgm-analysis/all-in-one-trial")
DEMIX_FILE = "955717e8-8726e21a.th"
DEMIX_SHA256 = "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def source_revision(folder):
    result = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def preflight(wav, runtime_root, model):
    if not wav.is_file():
        raise FileNotFoundError(wav)
    with wave.open(str(wav), "rb") as audio:
        if audio.getcomptype() != "NONE" or audio.getsampwidth() != 2:
            raise ValueError("Input must be uncompressed 16-bit PCM WAV.")
        if audio.getframerate() != 44100 or audio.getnchannels() not in (1, 2):
            raise ValueError("Input must be canonical 44100 Hz mono/stereo WAV.")
        duration = audio.getnframes() / audio.getframerate()
        if not 0 < duration <= 600.001:
            raise ValueError("Input duration must be in (0, 600] seconds.")
    required = [runtime_root / name for name in
                ("packages", "infer-source/src", "madmom-source", "demucs-source")]
    for item in required:
        if not item.is_dir():
            raise FileNotFoundError(f"Installed runtime directory missing: {item}")
    import tomllib
    config = runtime_root / "infer-source/src/allin1_infer/config/checkpoints.toml"
    manifest = tomllib.loads(config.read_text(encoding="utf-8"))
    if model not in manifest["models"]:
        raise ValueError(f"Unsupported cached model: {model}")
    checkpoint_dir = runtime_root / "torch-cache/hub/checkpoints"
    weights = []
    for artifact in manifest["models"][model]["artifacts"]:
        if artifact.get("kind") != "checkpoint":
            continue
        item = checkpoint_dir / artifact["name"]
        if not item.is_file():
            raise FileNotFoundError(f"Model not cached; no automatic downloads: {item}")
        actual = digest(item)
        if item.stat().st_size != artifact["size"] or actual != artifact["sha256"]:
            raise ValueError(f"Cached checkpoint identity mismatch: {item}")
        weights.append({"file": str(item), "sha256": actual, "bytes": item.stat().st_size})
    separator = checkpoint_dir / DEMIX_FILE
    if not separator.is_file():
        raise FileNotFoundError(f"HTDemucs not cached; no automatic downloads: {separator}")
    if separator.stat().st_size != 84141911 or digest(separator) != DEMIX_SHA256:
        raise ValueError(f"Cached HTDemucs identity mismatch: {separator}")
    weights.append({"file": str(separator), "sha256": DEMIX_SHA256,
                    "bytes": separator.stat().st_size})
    revisions = {name: source_revision(runtime_root / name) for name in
                 ("infer-source", "madmom-source", "demucs-source")}
    return duration, weights, revisions


def setup_runtime(runtime_root, threads):
    for name, suffix in (("TORCH_HOME", "torch-cache"), ("HF_HOME", "hf-cache"),
                         ("NUMBA_CACHE_DIR", "numba-cache"), ("MPLCONFIGDIR", "mpl-cache")):
        os.environ[name] = str(runtime_root / suffix)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    # Mirrors the installed runtime bootstrap; no system package changes.
    for path in (runtime_root / "packages", runtime_root / "infer-source/src",
                 runtime_root / "madmom-source", runtime_root / "demucs-source",
                 runtime_root.parent / "packages"):
        sys.path.insert(0, str(path))
    import torch
    torch.set_num_threads(threads)
    # A changed upstream loader must fail rather than download an unbudgeted model.
    def offline_download(*args, **kwargs):
        raise RuntimeError("Network model download disabled; populate verified local runtime first.")
    import urllib.request
    urllib.request.urlretrieve = offline_download
    torch.hub.download_url_to_file = offline_download
    return torch


def _guard_operation(stream, acquire):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)


def acquire_analysis_lock(cache):
    """Hold an OS-released gate plus inspectable owner record across inference.

    A killed worker releases the OS gate automatically. A subsequent worker
    may remove only its stale owner record/directory after checking PID and
    process creation time. The gate serializes stale-lock recovery.
    """
    import psutil
    guard = (cache / "guard.lock").open("a+b")
    if guard.seek(0, 2) == 0:
        guard.write(b"0")
        guard.flush()
    try:
        _guard_operation(guard, True)
    except (OSError, BlockingIOError):
        guard.close()
        raise RuntimeError(f"Same audio/model analysis is active: {cache / 'running.lock'}")
    lock = cache / "running.lock"
    try:
        if lock.exists():
            owner_file = lock / "owner.json"
            if not owner_file.is_file():
                raise RuntimeError(f"Unverifiable old lock without owner; inspect before removing: {lock}")
            try:
                owner = read_json(owner_file)
                pid, created = int(owner["pid"]), float(owner["create_time"])
                process = psutil.Process(pid)
                live = process.is_running() and abs(process.create_time() - created) < 0.01
            except psutil.NoSuchProcess:
                live = False
            except (KeyError, ValueError, TypeError, psutil.AccessDenied) as exc:
                raise RuntimeError(f"Cannot verify existing lock owner: {owner_file}") from exc
            if live:
                raise RuntimeError(f"Existing analysis owner is still alive: PID {pid}")
            # No recursive removal: unexpected contents remain for inspection.
            if {child.name for child in lock.iterdir()} != {"owner.json"}:
                raise RuntimeError(f"Unexpected stale lock contents; inspect: {lock}")
            owner_file.unlink()
            lock.rmdir()
        lock.mkdir()
        process = psutil.Process(os.getpid())
        (lock / "owner.json").write_text(json.dumps({"pid": process.pid,
            "create_time": process.create_time()}), encoding="utf-8")
        return guard
    except BaseException:
        _guard_operation(guard, False)
        guard.close()
        raise


def release_analysis_lock(cache, guard):
    lock = cache / "running.lock"
    try:
        (lock / "owner.json").unlink()
        lock.rmdir()
    finally:
        _guard_operation(guard, False)
        guard.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--model", default="harmonix-all",
                        choices=["harmonix-all"] + [f"harmonix-fold{i}" for i in range(8)])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional content-addressed cache; default under runtime-root.")
    parser.add_argument("--reuse-run", type=Path,
                        help="Recover completed stems from a stopped run with the same verified input WAV.")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8:
        parser.error("threads must be in [1,8].")
    wav = args.wav.resolve()
    out = args.output_dir.resolve()
    runtime = args.runtime_root.resolve()
    if out.exists():
        parser.error("Use a new output directory; existing results are never overwritten.")
    if shutil.which("git") is None:
        parser.error("git is required to identify installed source revisions.")
    started = time.perf_counter()
    duration, weights, revisions = preflight(wav, runtime, args.model)
    audio_hash = digest(wav)
    identity = {"wav_sha256": audio_hash, "model": args.model,
                "source_revisions": revisions, "weights_sha256": [w["sha256"] for w in weights],
                "worker_sha256": digest(Path(__file__)),
                "preparation_sha256": {name: digest(Path(__file__).with_name(name)) for name in ['prepare_stems.py', 'prepare_spectrograms.py']}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    cache = (args.cache_dir.resolve() if args.cache_dir else runtime / "analysis-cache") / key
    cache.mkdir(parents=True, exist_ok=True)
    guard = acquire_analysis_lock(cache)
    try:
        out.mkdir(parents=True, exist_ok=False)
        complete = cache / "complete.json"
        cache_hit = False
        if complete.is_file():
            previous = read_json(complete)
            candidate = Path(previous["raw_structure_file"])
            if previous["identity"] == identity and candidate.is_file() and digest(candidate) == previous["raw_sha256"]:
                run = Path(previous["run_directory"])
                raw = candidate
                cache_hit = True
        if not cache_hit:
            candidates = ([args.reuse_run.resolve()] if args.reuse_run else [])
            candidates += sorted(cache.glob('run-*'), key=lambda p: p.stat().st_mtime, reverse=True)
            run = None
            for candidate in candidates:
                local = candidate / 'input.wav'
                stems = candidate / 'demix/htdemucs/input'
                if not local.is_file() or digest(local) != audio_hash:
                    if args.reuse_run and candidate == args.reuse_run.resolve():
                        raise ValueError('Recovery run describes a different input WAV.')
                    continue
                valid = True
                for name in ('vocals', 'drums', 'bass', 'other'):
                    stem = stems / (name + '.wav')
                    try:
                        with wave.open(str(stem), 'rb') as handle:
                            valid = valid and abs(handle.getnframes()/handle.getframerate()-duration) < .05
                    except (OSError, wave.Error):
                        valid = False
                if valid:
                    run = candidate
                    break
            if run is None:
                run = cache / ("run-" + uuid.uuid4().hex)
                run.mkdir()
                # A content-specific copy prevents All-In-One's filename-based cache collisions.
                shutil.copy2(wav, run / 'input.wav')
            local_wav = run / 'input.wav'
            if digest(local_wav) != audio_hash:
                raise ValueError("Input changed while preparing structure analysis.")
            # Demucs tensors and its allocator used to remain resident during full-song
            # spectrogram extraction. An isolated process makes the release reliable.
            subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_stems.py')),
                            '--wav', str(local_wav), '--runtime-root', str(runtime),
                            '--demix-dir', str(run/'demix'), '--threads', str(args.threads)],
                           check=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_spectrograms.py')),
                            '--stems-dir', str(run/'demix/htdemucs/input'),
                            '--output', str(run/'spectrograms/input.npy'),
                            '--runtime-root', str(runtime), '--threads', str(args.threads)],
                           check=True)
            torch = setup_runtime(runtime, args.threads)
            import allin1_infer
            result = allin1_infer.analyze(
                local_wav, out_dir=run / "raw", model=args.model, device="cpu",
                demix_dir=run / "demix", spec_dir=run / "spectrograms",
                keep_byproducts=True, multiprocess=False,
            )
            raw = run / "raw/input.json"
            if not raw.is_file():
                raise RuntimeError("All-In-One did not write expected raw result.")
            completed = {"identity": identity, "raw_structure_file": str(raw),
                         "raw_sha256": digest(raw), "run_directory": str(run)}
            temporary = cache / "complete.json.tmp"
            temporary.write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(complete)
        data = read_json(raw)
        if not isinstance(data.get("segments"), list) or not isinstance(data.get("beats"), list):
            raise ValueError("Unexpected structure result format.")
        destination = out / "structure.json"
        shutil.copy2(raw, destination)
        receipt = {"status": "success", "input_wav": str(wav), "sha256": audio_hash,
                   "duration_seconds": duration, "model": args.model, "device": "cpu",
                   "threads": args.threads, "runtime_root": str(runtime),
                   "implementation": "openmirlab/all-in-one-infer; original official weights; third-party PyTorch port",
                   "source_revisions": revisions, "weights": weights, "cache_hit": cache_hit,
                   "cache_directory": str(cache), "run_directory": str(run),
                   "raw_structure_file": str(destination), "raw_sha256": digest(destination),
                   "demix_directory": str(run / "demix"), "spectrogram_directory": str(run / "spectrograms"),
                   "elapsed_seconds": round(time.perf_counter() - started, 3),
                   "raw_output_note": "Raw output unchanged; terminal frame labels/times may extend beyond WAV and need downstream validation."}
        (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"structure": str(destination), "receipt": str(out / "receipt.json"),
                          "cache_hit": cache_hit}, ensure_ascii=False))
    finally:
        release_analysis_lock(cache, guard)


if __name__ == "__main__":
    main()
