#!/usr/bin/env python3
"""Analyze test outputs to show trends and regressions.

This script scans test output directories, extracts timing and error information,
and shows trends across multiple runs.

Usage:
    # Show last 5 runs summary
    python scripts/analyze_test_runs.py

    # Compare specific tests
    python scripts/analyze_test_runs.py --test "test_smoke*" --runs 10

    # Export to JSON
    python scripts/analyze_test_runs.py --json > report.json

    # Export to CSV
    python scripts/analyze_test_runs.py --csv > report.csv

    # Show only errors/warnings
    python scripts/analyze_test_runs.py --errors-only

    # Filter by backend
    python scripts/analyze_test_runs.py --backend llm_dit

    # Filter by date range
    python scripts/analyze_test_runs.py --since 2026-01-18
"""

import argparse
import csv
import fnmatch
import io
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Output directory relative to project root
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "tests"


@dataclass
class TimingInfo:
    """Timing information for a test run."""

    total_seconds: float = 0.0
    text_encoder_seconds: float = 0.0
    transformer_seconds: float = 0.0
    vae_seconds: float = 0.0


@dataclass
class MemoryInfo:
    """Memory usage information."""

    peak_gb: float = 0.0
    text_encoder_gb: float = 0.0
    transformer_gb: float = 0.0
    vae_gb: float = 0.0


@dataclass
class GenerationParams:
    """Parameters used for generation."""

    frames: int = 0
    height: int = 0
    width: int = 0
    steps: int = 0
    guidance_scale: float = 0.0
    seed: int = 0


@dataclass
class TestRunResult:
    """Result from a single test run."""

    timestamp: str
    test_name: str
    backend: str
    status: str  # PASS, FAIL, SKIP, ERROR
    timing: TimingInfo = field(default_factory=TimingInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    params: GenerationParams = field(default_factory=GenerationParams)
    error_count: int = 0
    warning_count: int = 0
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    output_dir: str = ""


@dataclass
class SessionSummary:
    """Summary of a test session."""

    timestamp: str
    backend: str
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0


def parse_timestamp(ts: str) -> datetime:
    """Parse timestamp string to datetime."""
    try:
        # Format: YYYYMMDD_HHMMSS
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except ValueError:
        # Try other formats
        for fmt in ["%Y-%m-%d_%H%M%S", "%Y-%m-%d"]:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        return datetime.min


def extract_test_info_from_dirname(dirname: str) -> tuple[str, str]:
    """Extract test name and timestamp from directory name.

    Format: test_name_YYYYMMDD_HHMMSS
    Returns: (test_name, timestamp)
    """
    # Pattern: everything before the timestamp
    match = re.match(r"^(.+)_(\d{8}_\d{6})$", dirname)
    if match:
        return match.group(1), match.group(2)
    return dirname, ""


def parse_timing_from_log(log_content: str) -> TimingInfo:
    """Parse timing information from log content."""
    timing = TimingInfo()

    # Look for pytest duration
    match = re.search(r"in\s+([\d.]+)s", log_content)
    if match:
        timing.total_seconds = float(match.group(1))

    # Look for stage timings
    stage_patterns = [
        (r"Loading text encoder.*?(\d+\.?\d*)s", "text_encoder_seconds"),
        (r"text_encoder.*?(\d+\.?\d*)s", "text_encoder_seconds"),
        (r"Transformer.*?(\d+\.?\d*)s", "transformer_seconds"),
        (r"transformer.*?(\d+\.?\d*)s", "transformer_seconds"),
        (r"VAE.*?(\d+\.?\d*)s", "vae_seconds"),
        (r"vae.*?(\d+\.?\d*)s", "vae_seconds"),
    ]

    for pattern, attr in stage_patterns:
        match = re.search(pattern, log_content, re.IGNORECASE)
        if match and getattr(timing, attr) == 0.0:
            setattr(timing, attr, float(match.group(1)))

    return timing


def parse_memory_from_log(log_content: str) -> MemoryInfo:
    """Parse memory usage information from log content."""
    memory = MemoryInfo()

    # Look for memory patterns
    patterns = [
        (r"Memory.*?(\d+\.?\d*)\s*GB", "peak_gb"),
        (r"VRAM.*?(\d+\.?\d*)\s*GB", "peak_gb"),
        (r"→\s*(\d+\.?\d*)GB\s*\(fp8", "peak_gb"),  # From loader
    ]

    for pattern, attr in patterns:
        match = re.search(pattern, log_content)
        if match and getattr(memory, attr) == 0.0:
            setattr(memory, attr, float(match.group(1)))

    return memory


def count_errors_warnings(log_content: str) -> tuple[int, int, list, list]:
    """Count errors and warnings in log content."""
    errors = []
    warnings = []

    for line in log_content.split("\n"):
        line_lower = line.lower()
        if "error" in line_lower and "no error" not in line_lower:
            # Skip non-error lines
            if any(x in line_lower for x in ["error_count", "errors.log", "errors:"]):
                continue
            errors.append(line.strip())
        elif "warning" in line_lower:
            # Skip expected warnings
            if "userwarning" in line_lower or "deprecationwarning" in line_lower:
                continue
            warnings.append(line.strip())

    return len(errors), len(warnings), errors, warnings


def parse_metadata(metadata_path: Path) -> GenerationParams:
    """Parse generation parameters from metadata.json."""
    params = GenerationParams()

    if not metadata_path.exists():
        return params

    try:
        with open(metadata_path) as f:
            data = json.load(f)

        params.frames = data.get("num_frames", 0)
        params.height = data.get("height", 0)
        params.width = data.get("width", 0)
        params.steps = data.get("num_inference_steps", 0)
        params.guidance_scale = data.get("guidance_scale", 0.0)
        params.seed = data.get("seed", 0)
    except (json.JSONDecodeError, OSError):
        pass

    return params


def scan_session_runs(outputs_dir: Path) -> list[SessionSummary]:
    """Scan session run directories."""
    sessions = []
    runs_dir = outputs_dir / "runs"

    if not runs_dir.exists():
        return sessions

    for session_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue

        summary = SessionSummary(timestamp=session_dir.name, backend="unknown")

        # Read environment.json
        env_path = session_dir / "environment.json"
        if env_path.exists():
            try:
                with open(env_path) as f:
                    env = json.load(f)
                summary.backend = env.get("backend", "unknown")
                summary.gpu_name = env.get("gpu_name", "")
                summary.gpu_vram_gb = env.get("gpu_vram_gb", 0.0)
            except (json.JSONDecodeError, OSError):
                pass

        # Read summary.json
        summary_path = session_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    data = json.load(f)
                summary.total_tests = data.get("total_tests", 0)
                summary.passed = data.get("passed", 0)
                summary.failed = data.get("failed", 0)
                summary.skipped = data.get("skipped", 0)
            except (json.JSONDecodeError, OSError):
                pass

        # Read session.log for timing
        session_log = session_dir / "session.log"
        if session_log.exists():
            try:
                content = session_log.read_text()
                timing = parse_timing_from_log(content)
                summary.duration_seconds = timing.total_seconds
            except OSError:
                pass

        sessions.append(summary)

    return sessions


def scan_test_outputs(
    outputs_dir: Path, backend_filter: Optional[str] = None
) -> list[TestRunResult]:
    """Scan all test output directories."""
    results = []

    # Scan backend directories (ltx2, llm_dit, etc.)
    for backend_dir in outputs_dir.iterdir():
        if not backend_dir.is_dir():
            continue
        if backend_dir.name in ["runs", ".gitkeep"]:
            continue

        backend_name = backend_dir.name

        if backend_filter and backend_filter != backend_name:
            continue

        # Scan test output directories
        for test_dir in sorted(backend_dir.iterdir(), reverse=True):
            if not test_dir.is_dir():
                continue

            test_name, timestamp = extract_test_info_from_dirname(test_dir.name)
            if not timestamp:
                continue

            result = TestRunResult(
                timestamp=timestamp,
                test_name=test_name,
                backend=backend_name,
                status="PASS",  # Default, will be updated
                output_dir=str(test_dir),
            )

            # Parse metadata.json
            metadata_path = test_dir / "metadata.json"
            result.params = parse_metadata(metadata_path)

            # Parse various log files
            for log_file in ["generation.log", "debug.log", "session.log"]:
                log_path = test_dir / log_file
                if log_path.exists():
                    try:
                        content = log_path.read_text()
                        timing = parse_timing_from_log(content)
                        if timing.total_seconds > result.timing.total_seconds:
                            result.timing = timing

                        memory = parse_memory_from_log(content)
                        if memory.peak_gb > result.memory.peak_gb:
                            result.memory = memory
                    except OSError:
                        pass

            # Parse errors.log
            errors_log = test_dir / "errors.log"
            if errors_log.exists():
                try:
                    content = errors_log.read_text()
                    err_count, warn_count, errs, warns = count_errors_warnings(content)
                    result.error_count = err_count
                    result.warning_count = warn_count
                    result.errors = errs[:10]  # Keep only first 10
                    result.warnings = warns[:10]

                    if err_count > 0:
                        result.status = "ERROR"
                except OSError:
                    pass

            results.append(result)

    # Also scan baseline directory
    baseline_dir = outputs_dir / "baseline"
    if baseline_dir.exists():
        for backend_dir in baseline_dir.iterdir():
            if not backend_dir.is_dir():
                continue

            backend_name = backend_dir.name

            if backend_filter and backend_filter != backend_name:
                continue

            for test_dir in sorted(backend_dir.iterdir(), reverse=True):
                if not test_dir.is_dir():
                    continue

                test_name, timestamp = extract_test_info_from_dirname(test_dir.name)
                if not timestamp:
                    continue

                result = TestRunResult(
                    timestamp=timestamp,
                    test_name=test_name,
                    backend=backend_name,
                    status="PASS",
                    output_dir=str(test_dir),
                )

                # Parse various json files
                for json_file in test_dir.glob("*.json"):
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                        if "num_frames" in data:
                            result.params.frames = data.get("num_frames", 0)
                        if "height" in data:
                            result.params.height = data.get("height", 0)
                        if "width" in data:
                            result.params.width = data.get("width", 0)
                    except (json.JSONDecodeError, OSError):
                        pass

                results.append(result)

    return results


def filter_results(
    results: list[TestRunResult],
    test_pattern: Optional[str] = None,
    since_date: Optional[str] = None,
    max_runs: int = 5,
) -> list[TestRunResult]:
    """Filter results by test pattern, date, and limit."""
    filtered = results

    # Filter by test pattern
    if test_pattern:
        filtered = [r for r in filtered if fnmatch.fnmatch(r.test_name, test_pattern)]

    # Filter by date
    if since_date:
        try:
            since_dt = datetime.strptime(since_date, "%Y-%m-%d")
            filtered = [r for r in filtered if parse_timestamp(r.timestamp) >= since_dt]
        except ValueError:
            pass

    # Sort by timestamp (newest first) and limit per test
    test_runs = defaultdict(list)
    for r in filtered:
        test_runs[r.test_name].append(r)

    final_results = []
    for test_name, runs in test_runs.items():
        runs.sort(key=lambda x: x.timestamp, reverse=True)
        final_results.extend(runs[:max_runs])

    return final_results


def calculate_trends(
    results: list[TestRunResult],
) -> dict[str, dict]:
    """Calculate trends across runs for each test."""
    trends = {}

    # Group by test name
    by_test = defaultdict(list)
    for r in results:
        by_test[r.test_name].append(r)

    for test_name, runs in by_test.items():
        if len(runs) < 2:
            trends[test_name] = {"insufficient_data": True}
            continue

        # Sort by timestamp (oldest first for trend calculation)
        runs.sort(key=lambda x: x.timestamp)

        # Calculate deltas between first and last
        first, last = runs[0], runs[-1]

        def calc_delta(old: float, new: float) -> tuple[float, str]:
            if old == 0:
                return 0.0, ""
            delta = new - old
            pct = (delta / old) * 100
            if abs(pct) < 1:
                return delta, "stable"
            return delta, "improved" if delta < 0 else "regressed"

        total_delta, total_trend = calc_delta(
            first.timing.total_seconds, last.timing.total_seconds
        )
        encoder_delta, encoder_trend = calc_delta(
            first.timing.text_encoder_seconds, last.timing.text_encoder_seconds
        )
        transformer_delta, transformer_trend = calc_delta(
            first.timing.transformer_seconds, last.timing.transformer_seconds
        )
        vae_delta, vae_trend = calc_delta(
            first.timing.vae_seconds, last.timing.vae_seconds
        )
        error_delta = last.error_count - first.error_count
        error_trend = (
            "improved"
            if error_delta < 0
            else "regressed" if error_delta > 0 else "stable"
        )

        # Pass rate
        pass_count = sum(1 for r in runs if r.status == "PASS")
        pass_rate = (pass_count / len(runs)) * 100

        trends[test_name] = {
            "num_runs": len(runs),
            "pass_rate": pass_rate,
            "timing": {
                "total_delta": total_delta,
                "total_trend": total_trend,
                "encoder_delta": encoder_delta,
                "encoder_trend": encoder_trend,
                "transformer_delta": transformer_delta,
                "transformer_trend": transformer_trend,
                "vae_delta": vae_delta,
                "vae_trend": vae_trend,
            },
            "error_delta": error_delta,
            "error_trend": error_trend,
            "avg_total_time": sum(r.timing.total_seconds for r in runs) / len(runs),
        }

    return trends


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds == 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.1f}s"


def format_delta(delta: float, is_time: bool = True) -> str:
    """Format delta with arrow indicator."""
    if delta == 0:
        return "-"
    sign = "+" if delta > 0 else ""
    if is_time:
        # For time, negative is good
        indicator = "[green]v[/green]" if delta < 0 else "[red]^[/red]"
        return f"{sign}{delta:.1f}s {indicator}"
    else:
        # For errors, negative is good
        indicator = "[green]v[/green]" if delta < 0 else "[red]^[/red]"
        return f"{sign}{delta} {indicator}"


def format_trend_arrow(trend: str) -> str:
    """Format trend as arrow."""
    if trend == "improved":
        return "[green]v[/green]"
    elif trend == "regressed":
        return "[red]^[/red]"
    return "-"


def print_table(headers: list[str], rows: list[list[str]], col_widths: list[int]):
    """Print a simple ASCII table."""
    # Build format string
    fmt = " ".join(f"{{:<{w}}}" for w in col_widths)

    # Print header
    print(fmt.format(*headers))
    print("-" * sum(col_widths + [len(col_widths) - 1]))

    # Print rows
    for row in rows:
        # Truncate long values
        truncated = []
        for val, width in zip(row, col_widths):
            if len(str(val)) > width:
                truncated.append(str(val)[: width - 2] + "..")
            else:
                truncated.append(str(val))
        print(fmt.format(*truncated))


def output_cli(
    results: list[TestRunResult],
    sessions: list[SessionSummary],
    trends: dict[str, dict],
    errors_only: bool = False,
):
    """Output results in CLI table format."""
    # Try to import tabulate for nicer formatting
    try:
        from tabulate import tabulate

        has_tabulate = True
    except ImportError:
        has_tabulate = False

    # Header
    print()
    print("LTX-2 Test Analysis Report")
    print("=" * 60)
    print()

    # Environment info from most recent session
    if sessions:
        latest = sessions[0]
        print(f"Environment: {latest.backend} backend, {latest.gpu_name}")
        if latest.gpu_vram_gb:
            print(f"GPU VRAM: {latest.gpu_vram_gb:.1f}GB")
        print()

    # Get unique tests
    tests = sorted(set(r.test_name for r in results))

    for test_name in tests:
        test_results = [r for r in results if r.test_name == test_name]
        test_results.sort(key=lambda x: x.timestamp, reverse=True)

        # Skip if no errors and errors_only mode
        if errors_only and all(r.error_count == 0 for r in test_results):
            continue

        print(f"Test: {test_name}")
        print("-" * 60)

        # Build table data
        headers = ["Run", "Total", "Encoder", "Transformer", "VAE", "Status", "Errors"]
        rows = []

        for r in test_results:
            status = r.status
            if status == "ERROR":
                status_str = "FAIL"
            elif status == "SKIP":
                status_str = "SKIP"
            else:
                status_str = "PASS"

            error_str = str(r.error_count) if r.error_count > 0 else "0"
            if r.warning_count > 0:
                error_str += f" ({r.warning_count}w)"

            rows.append(
                [
                    r.timestamp,
                    format_time(r.timing.total_seconds),
                    format_time(r.timing.text_encoder_seconds),
                    format_time(r.timing.transformer_seconds),
                    format_time(r.timing.vae_seconds),
                    status_str,
                    error_str,
                ]
            )

        # Print table
        if has_tabulate:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            col_widths = [18, 8, 10, 12, 8, 8, 10]
            print_table(headers, rows, col_widths)

        # Print trend if available
        if test_name in trends and not trends[test_name].get("insufficient_data"):
            t = trends[test_name]
            print()
            print("Trend:", end=" ")
            parts = []
            if t["timing"]["total_delta"] != 0:
                delta = t["timing"]["total_delta"]
                arrow = "v" if delta < 0 else "^"
                parts.append(f"Total: {delta:+.1f}s {arrow}")
            if t["error_delta"] != 0:
                arrow = "v" if t["error_delta"] < 0 else "^"
                parts.append(f"Errors: {t['error_delta']:+d} {arrow}")
            if parts:
                print(" | ".join(parts))
            else:
                print("stable")

        print()

    # Summary section
    print("Summary")
    print("-" * 60)

    # Calculate overall stats
    total_tests = len(tests)
    total_runs = len(results)
    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status in ("FAIL", "ERROR"))
    skip_count = sum(1 for r in results if r.status == "SKIP")

    print(f"Tests analyzed: {total_tests}")
    print(f"Total runs: {total_runs}")
    if total_runs > 0:
        print(f"Pass rate: {pass_count/total_runs*100:.0f}% ({pass_count}/{total_runs})")

    # Time range
    if results:
        timestamps = [parse_timestamp(r.timestamp) for r in results]
        min_ts = min(ts for ts in timestamps if ts != datetime.min)
        max_ts = max(ts for ts in timestamps if ts != datetime.min)
        print(f"Date range: {min_ts.strftime('%Y-%m-%d')} to {max_ts.strftime('%Y-%m-%d')}")

    # Overall trends
    avg_errors = sum(r.error_count for r in results) / max(1, len(results))
    print(f"Avg errors per run: {avg_errors:.1f}")

    # Show any recurring errors
    all_errors = []
    for r in results:
        all_errors.extend(r.errors)
    if all_errors and not errors_only:
        print()
        print("Recent errors:")
        for err in all_errors[:5]:
            print(f"  - {err[:80]}...")

    print()


def output_json(
    results: list[TestRunResult],
    sessions: list[SessionSummary],
    trends: dict[str, dict],
):
    """Output results as JSON."""
    # Convert dataclasses to dicts
    output = {
        "sessions": [asdict(s) for s in sessions],
        "results": [asdict(r) for r in results],
        "trends": trends,
        "summary": {
            "total_tests": len(set(r.test_name for r in results)),
            "total_runs": len(results),
            "pass_count": sum(1 for r in results if r.status == "PASS"),
            "fail_count": sum(1 for r in results if r.status in ("FAIL", "ERROR")),
            "skip_count": sum(1 for r in results if r.status == "SKIP"),
        },
    }

    print(json.dumps(output, indent=2, default=str))


def output_csv(results: list[TestRunResult]):
    """Output results as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(
        [
            "timestamp",
            "test_name",
            "backend",
            "status",
            "total_seconds",
            "encoder_seconds",
            "transformer_seconds",
            "vae_seconds",
            "peak_memory_gb",
            "error_count",
            "warning_count",
            "frames",
            "height",
            "width",
            "steps",
            "guidance_scale",
            "seed",
        ]
    )

    # Data rows
    for r in results:
        writer.writerow(
            [
                r.timestamp,
                r.test_name,
                r.backend,
                r.status,
                r.timing.total_seconds,
                r.timing.text_encoder_seconds,
                r.timing.transformer_seconds,
                r.timing.vae_seconds,
                r.memory.peak_gb,
                r.error_count,
                r.warning_count,
                r.params.frames,
                r.params.height,
                r.params.width,
                r.params.steps,
                r.params.guidance_scale,
                r.params.seed,
            ]
        )

    print(output.getvalue())


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test outputs to show trends and regressions."
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=5,
        help="Number of recent runs to analyze per test (default: 5)",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        help="Filter by test name pattern (supports wildcards, e.g., 'test_smoke*')",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=["llm_dit", "ltx2"],
        help="Filter by backend",
    )
    parser.add_argument(
        "--since",
        "-s",
        type=str,
        help="Show only runs since date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--csv",
        "-c",
        action="store_true",
        help="Output as CSV",
    )
    parser.add_argument(
        "--errors-only",
        "-e",
        action="store_true",
        help="Show only tests with errors/warnings",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(OUTPUTS_DIR),
        help=f"Test outputs directory (default: {OUTPUTS_DIR})",
    )

    args = parser.parse_args()

    outputs_dir = Path(args.output_dir)
    if not outputs_dir.exists():
        print(f"Error: Output directory not found: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    # Scan data
    sessions = scan_session_runs(outputs_dir)
    results = scan_test_outputs(outputs_dir, backend_filter=args.backend)

    if not results:
        print("No test results found.", file=sys.stderr)
        sys.exit(0)

    # Filter results
    results = filter_results(
        results,
        test_pattern=args.test,
        since_date=args.since,
        max_runs=args.runs,
    )

    # Calculate trends
    trends = calculate_trends(results)

    # Output
    if args.json:
        output_json(results, sessions, trends)
    elif args.csv:
        output_csv(results)
    else:
        output_cli(results, sessions, trends, errors_only=args.errors_only)


if __name__ == "__main__":
    main()
