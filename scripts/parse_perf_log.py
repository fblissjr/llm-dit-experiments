#!/usr/bin/env python3
"""
Parse LTX-2 generation timing from structured JSONL server logs.

Extracts per-generation timing breakdowns from logs/llm_dit.jsonl,
groups into generation sessions, and prints a summary table. Useful for
tracking performance across code changes and config tweaks.

Last updated: 2026-03-08

Usage:
    uv run python scripts/parse_perf_log.py                     # latest 5 runs
    uv run python scripts/parse_perf_log.py -n 20               # latest 20 runs
    uv run python scripts/parse_perf_log.py --all                # all runs
    uv run python scripts/parse_perf_log.py --log path/to.jsonl  # custom log
    uv run python scripts/parse_perf_log.py --csv                # CSV output
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import orjson as json

    def loads(b):
        return json.loads(b)

except ImportError:
    import json  # type: ignore[no-redef]

    def loads(b):
        return json.loads(b)


DEFAULT_LOG = Path(__file__).parent.parent / "logs" / "llm_dit.jsonl"

# Regex patterns for extracting timing from log messages
RE_STAGE_COMPLETE = re.compile(
    r"Stage (\d+(?:\.\d+)?) complete: ([\d.]+)s"
)
RE_DENOISE_SUMMARY = re.compile(
    r"\[(\w+)] (\d+) steps in ([\d.]+)s"
)
RE_STEP_TIME = re.compile(
    r"\[(\w+):Step (\d+)] ([\d.]+)s"
)
RE_VAE_DECODE = re.compile(
    r"\[Decode] VAE decode ([\d.]+)s"
)
RE_VRAM = re.compile(
    r"\[VRAM:(\w+)] allocated=([\d.]+)GB, reserved=([\d.]+)GB, cuda_free=([\d.]+)GB"
)
RE_LORA_FUSE = re.compile(
    r"Fused (\d+) LoRA layers \(alpha=([\d.]+)\)"
)
RE_AUDIO_DECODE = re.compile(
    r"Stage 4: Decoding audio"
)
RE_LATENT_STATS = re.compile(
    r"\[(\w+):Step 0] video latent std=([\d.]+), audio latent std=([\d.]+)"
)
RE_SIGMA_DEBUG = re.compile(
    r"Stage (\d) sigmas: \[([\d.]+) -> ([\d.]+)], (\d+) steps, mode=(\w+)"
)
RE_ENCODER_LOAD = re.compile(
    r"Stage 0: Loading text encoder"
)
RE_TOTAL_PIPELINE = re.compile(
    r"Total pipeline: ([\d.]+)s"
)


@dataclass
class GenerationRun:
    """Timing data for a single generation run."""

    timestamp: str = ""
    stage0_s: float = 0.0       # Encoding
    stage1_s: float = 0.0       # Low-res denoise
    stage1_steps: int = 0
    stage1_denoise_s: float = 0.0
    stage1_step0_s: float = 0.0
    stage15_s: float = 0.0      # Spatial upsample
    stage2_s: float = 0.0       # High-res denoise
    stage2_steps: int = 0
    stage2_denoise_s: float = 0.0
    stage2_step0_s: float = 0.0
    stage3_s: float = 0.0       # VAE decode
    vae_decode_s: float = 0.0
    has_audio: bool = False
    lora_fuse_layers: int = 0
    total_s: float = 0.0
    mode: str = ""              # AV or video-only
    vram_peak_gb: float = 0.0
    # Debug diagnostics (--debug only)
    video_latent_std_step0: float = 0.0
    audio_latent_std_step0: float = 0.0
    sigma_start: float = 0.0
    sigma_end: float = 0.0
    # Step-level detail
    step_times: dict[str, list[float]] = field(default_factory=dict)

    @property
    def total_computed(self) -> float:
        """Sum of all stages (use when total_s not logged)."""
        return self.stage0_s + self.stage1_s + self.stage15_s + self.stage2_s + self.stage3_s

    @property
    def overhead_s(self) -> float:
        """Time outside denoising (loading, offloading, LoRA fusion)."""
        return self.total_computed - self.stage1_denoise_s - self.stage2_denoise_s - self.vae_decode_s

    @property
    def step_avg_s(self) -> float:
        """Average time per denoising step across both stages."""
        total_steps = self.stage1_steps + self.stage2_steps
        total_denoise = self.stage1_denoise_s + self.stage2_denoise_s
        return total_denoise / total_steps if total_steps > 0 else 0.0


def parse_log(log_path: Path) -> list[GenerationRun]:
    """Parse JSONL log and extract generation runs."""
    runs: list[GenerationRun] = []
    current: GenerationRun | None = None
    max_vram_alloc = 0.0

    with open(log_path, "rb") as f:
        for line_bytes in f:
            line_bytes = line_bytes.strip()
            if not line_bytes:
                continue
            try:
                entry = loads(line_bytes)
            except Exception:
                continue

            msg = entry.get("message", "")
            ts = entry.get("timestamp", "")
            logger = entry.get("logger", "")

            # Only look at generate.py and related loggers
            if logger not in (
                "llm_dit.pipelines.generate",
                "llm_dit.utils.memory",
                "llm_dit.utils.lora",
            ):
                continue

            # Start of a new generation
            if RE_ENCODER_LOAD.search(msg):
                # Save previous run if it has data
                if current and current.stage0_s > 0:
                    current.total_s = current.total_s or current.total_computed
                    current.vram_peak_gb = max_vram_alloc
                    runs.append(current)
                current = GenerationRun(timestamp=ts)
                max_vram_alloc = 0.0
                continue

            if current is None:
                continue

            # Stage completion times
            m = RE_STAGE_COMPLETE.search(msg)
            if m:
                stage, elapsed = m.group(1), float(m.group(2))
                if stage == "0":
                    current.stage0_s = elapsed
                elif stage == "1":
                    current.stage1_s = elapsed
                elif stage == "1.5":
                    current.stage15_s = elapsed
                elif stage == "2":
                    current.stage2_s = elapsed
                elif stage == "3":
                    current.stage3_s = elapsed
                continue

            # Denoise summary (e.g. "[stage1_denoise] 30 steps in 18.5s")
            m = RE_DENOISE_SUMMARY.search(msg)
            if m:
                stage_name, steps, elapsed = m.group(1), int(m.group(2)), float(m.group(3))
                if "stage1" in stage_name:
                    current.stage1_steps = steps
                    current.stage1_denoise_s = elapsed
                elif "stage2" in stage_name:
                    current.stage2_steps = steps
                    current.stage2_denoise_s = elapsed
                continue

            # Individual step times
            m = RE_STEP_TIME.search(msg)
            if m:
                stage_name, step_idx, elapsed = m.group(1), int(m.group(2)), float(m.group(3))
                if stage_name not in current.step_times:
                    current.step_times[stage_name] = []
                current.step_times[stage_name].append(elapsed)
                if step_idx == 0:
                    if "stage1" in stage_name:
                        current.stage1_step0_s = elapsed
                    elif "stage2" in stage_name:
                        current.stage2_step0_s = elapsed
                continue

            # VAE decode
            m = RE_VAE_DECODE.search(msg)
            if m:
                current.vae_decode_s = float(m.group(1))
                continue

            # VRAM snapshots
            m = RE_VRAM.search(msg)
            if m:
                alloc_gb = float(m.group(2))
                max_vram_alloc = max(max_vram_alloc, alloc_gb)
                continue

            # LoRA fusion
            m = RE_LORA_FUSE.search(msg)
            if m:
                current.lora_fuse_layers = int(m.group(1))
                continue

            # Audio stage
            if RE_AUDIO_DECODE.search(msg):
                current.has_audio = True
                continue

            # Latent stats (debug)
            m = RE_LATENT_STATS.search(msg)
            if m:
                current.video_latent_std_step0 = float(m.group(2))
                current.audio_latent_std_step0 = float(m.group(3))
                continue

            # Sigma debug
            m = RE_SIGMA_DEBUG.search(msg)
            if m:
                current.sigma_start = float(m.group(2))
                current.sigma_end = float(m.group(3))
                current.mode = m.group(5)
                continue

            # Total pipeline
            m = RE_TOTAL_PIPELINE.search(msg)
            if m:
                current.total_s = float(m.group(1))
                continue

    # Don't forget the last run
    if current and current.stage0_s > 0:
        current.total_s = current.total_s or current.total_computed
        current.vram_peak_gb = max_vram_alloc
        runs.append(current)

    return runs


def print_table(runs: list[GenerationRun]) -> None:
    """Print a formatted timing table."""
    if not runs:
        print("No generation runs found in log.")
        return

    # Header
    print()
    print(f"{'Timestamp':<20} {'Mode':<6} {'Encode':>7} {'S1 Dn':>7} {'S1 Tot':>7} "
          f"{'Upsamp':>7} {'S2 Dn':>7} {'S2 Tot':>7} {'Decode':>7} {'Total':>7} "
          f"{'Ovrhd':>7} {'Step':>6} {'VRAM':>6}")
    print("-" * 118)

    for r in runs:
        ts_short = r.timestamp[5:16] if len(r.timestamp) >= 16 else r.timestamp
        mode = r.mode or ("AV" if r.has_audio else "V")
        total = r.total_s or r.total_computed

        print(
            f"{ts_short:<20} {mode:<6} "
            f"{r.stage0_s:>6.1f}s {r.stage1_denoise_s:>6.1f}s {r.stage1_s:>6.1f}s "
            f"{r.stage15_s:>6.1f}s {r.stage2_denoise_s:>6.1f}s {r.stage2_s:>6.1f}s "
            f"{r.vae_decode_s:>6.1f}s {total:>6.1f}s "
            f"{r.overhead_s:>6.1f}s {r.step_avg_s:>5.3f}s "
            f"{r.vram_peak_gb:>5.1f}G"
        )

    # Summary stats if multiple runs
    if len(runs) > 1:
        print("-" * 118)
        totals = [r.total_s or r.total_computed for r in runs]
        step_avgs = [r.step_avg_s for r in runs if r.step_avg_s > 0]

        print(
            f"{'AVERAGE':<20} {'':6} "
            f"{sum(r.stage0_s for r in runs)/len(runs):>6.1f}s "
            f"{sum(r.stage1_denoise_s for r in runs)/len(runs):>6.1f}s "
            f"{sum(r.stage1_s for r in runs)/len(runs):>6.1f}s "
            f"{sum(r.stage15_s for r in runs)/len(runs):>6.1f}s "
            f"{sum(r.stage2_denoise_s for r in runs)/len(runs):>6.1f}s "
            f"{sum(r.stage2_s for r in runs)/len(runs):>6.1f}s "
            f"{sum(r.vae_decode_s for r in runs)/len(runs):>6.1f}s "
            f"{sum(totals)/len(totals):>6.1f}s "
            f"{sum(r.overhead_s for r in runs)/len(runs):>6.1f}s "
            f"{sum(step_avgs)/len(step_avgs):>5.3f}s "
            f"{'':>6}"
        )
        print(
            f"{'MIN/MAX':<20} {'':6} "
            f"{'':>7} {'':>7} {'':>7} {'':>7} {'':>7} {'':>7} {'':>7} "
            f"{min(totals):>3.0f}/{max(totals):<3.0f}s "
            f"{'':>7} "
            f"{min(step_avgs):>.3f}/{max(step_avgs):<.3f}"
            f"{'':>1}"
        )

    print()

    # Per-step breakdown for latest run (if available)
    latest = runs[-1]
    if latest.step_times:
        print("Latest run step-level detail:")
        for stage_name, times in latest.step_times.items():
            if len(times) <= 3:
                detail = ", ".join(f"{t:.2f}s" for t in times)
            else:
                detail = (
                    f"first={times[0]:.2f}s, "
                    f"avg={sum(times)/len(times):.3f}s, "
                    f"last={times[-1]:.2f}s"
                )
            print(f"  {stage_name}: {len(times)} steps logged -- {detail}")

    # Debug diagnostics from latest run
    if latest.video_latent_std_step0 > 0:
        print(f"\nLatest run diagnostics (step 0):")
        print(f"  Video latent std: {latest.video_latent_std_step0:.4f}")
        if latest.audio_latent_std_step0 > 0:
            print(f"  Audio latent std: {latest.audio_latent_std_step0:.4f}")
    if latest.sigma_start > 0:
        print(f"  Sigma range: {latest.sigma_start:.4f} -> {latest.sigma_end:.4f}")


def print_csv(runs: list[GenerationRun]) -> None:
    """Print CSV output for further analysis."""
    print("timestamp,mode,encode_s,s1_denoise_s,s1_total_s,upsample_s,"
          "s2_denoise_s,s2_total_s,vae_decode_s,total_s,overhead_s,"
          "step_avg_s,vram_peak_gb,s1_steps,s2_steps")
    for r in runs:
        total = r.total_s or r.total_computed
        mode = r.mode or ("AV" if r.has_audio else "V")
        print(
            f"{r.timestamp},{mode},{r.stage0_s:.2f},{r.stage1_denoise_s:.2f},"
            f"{r.stage1_s:.2f},{r.stage15_s:.2f},{r.stage2_denoise_s:.2f},"
            f"{r.stage2_s:.2f},{r.vae_decode_s:.2f},{total:.2f},"
            f"{r.overhead_s:.2f},{r.step_avg_s:.3f},{r.vram_peak_gb:.1f},"
            f"{r.stage1_steps},{r.stage2_steps}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Parse LTX-2 generation timing from server JSONL logs"
    )
    parser.add_argument(
        "--log", type=Path, default=DEFAULT_LOG,
        help=f"Path to JSONL log file (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "-n", type=int, default=5,
        help="Show latest N runs (default: 5)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show all runs",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Output as CSV instead of table",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    runs = parse_log(args.log)

    if not args.all:
        runs = runs[-args.n:]

    if args.csv:
        print_csv(runs)
    else:
        print(f"Parsed {len(runs)} generation run(s) from {args.log.name}")
        print_table(runs)


if __name__ == "__main__":
    main()
