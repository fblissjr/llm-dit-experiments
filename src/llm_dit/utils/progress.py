"""
Progress Tracking Utilities for llm-dit.

Last Updated: 2026-01-23

IMPORTANT: This module is PURE PYTORCH only.
Do NOT import or use any diffusers components.

Provides professional progress displays for:
- Denoising steps during generation
- Stage-based loading (text encoder, transformer, VAE)
- Batch processing

Ported from LTX-2 trainer's StandaloneSamplingProgress pattern.
Uses 'rich' library for beautiful terminal output.

Usage:
    # For denoising loop
    with SamplingProgress(num_steps=50, desc="Generating") as progress:
        for i, sigma in enumerate(sigmas[:-1]):
            # ... denoising step ...
            progress.advance()

    # Simple step tracker (no dependencies)
    tracker = StepTracker(total=50, desc="Denoising")
    for i in range(50):
        # ... step ...
        tracker.step()
    tracker.close()
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class StepTracker:
    """
    Simple step progress tracker without external dependencies.

    Uses basic print statements for progress. Useful as a fallback
    when rich is not available or for simple logging.

    Args:
        total: Total number of steps.
        desc: Description shown in progress output.
        log_interval: How often to log (every N steps, 0 = only start/end).

    Example:
        tracker = StepTracker(total=50, desc="Denoising")
        for i in range(50):
            # ... step ...
            tracker.step()
        tracker.close()
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        log_interval: int = 10,
    ):
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.start_time: Optional[float] = None
        self._last_log_step = -1

    def step(self) -> None:
        """Advance by one step."""
        if self.start_time is None:
            self.start_time = time.time()
            logger.info(f"{self.desc}: Starting ({self.total} steps)")

        self.current += 1

        # Log at intervals
        if self.log_interval > 0 and self.current % self.log_interval == 0:
            self._log_progress()
        elif self.current == self.total:
            self._log_progress()

    def _log_progress(self) -> None:
        """Log current progress."""
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0

        logger.info(
            f"{self.desc}: step {self.current}/{self.total} "
            f"({elapsed:.1f}s elapsed, {remaining:.1f}s remaining, {rate:.2f} it/s)"
        )
        self._last_log_step = self.current

    def close(self) -> None:
        """Finalize and log completion."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            logger.info(f"{self.desc}: Complete in {elapsed:.1f}s ({rate:.2f} it/s average)")


class SamplingProgress:
    """
    Professional progress display for sampling/denoising using rich.

    Provides a beautiful terminal progress bar with:
    - Step counter (step X/Y)
    - Progress bar visualization
    - Elapsed time
    - ETA (estimated time remaining)
    - Iteration rate (it/s)

    Falls back to StepTracker if rich is not available.

    Args:
        num_steps: Total number of denoising steps.
        desc: Description shown in progress bar (default: "Generating").
        disable: If True, disables progress display entirely.
        leave: If True, keeps progress bar visible after completion.

    Example:
        with SamplingProgress(num_steps=50, desc="Generating") as progress:
            for i in range(50):
                # ... denoising step ...
                progress.advance()

    Output:
        Generating [████████████████████░░░░░░░░░░░░░░░░░░░░] step 25/50 0:15 ETA: 0:15
    """

    def __init__(
        self,
        num_steps: int,
        desc: str = "Generating",
        disable: bool = False,
        leave: bool = True,
    ):
        self._num_steps = num_steps
        self._desc = desc
        self._disable = disable
        self._leave = leave

        # Rich components (created on __enter__)
        self._progress = None
        self._task = None
        self._console = None
        self._use_rich = False

        # Fallback tracker
        self._fallback: Optional[StepTracker] = None

    def __enter__(self) -> "SamplingProgress":
        if self._disable:
            return self

        # Try to use rich for beautiful progress
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._console = Console()
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40, style="blue"),
                TextColumn("{task.fields[info]}", style="cyan"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(compact=True),
                console=self._console,
                transient=not self._leave,
            )
            self._progress.start()
            self._task = self._progress.add_task(
                self._desc,
                total=self._num_steps,
                info=f"step 0/{self._num_steps}",
            )
            self._use_rich = True

        except ImportError:
            # Fall back to simple logging
            logger.debug("rich not available, using simple progress logging")
            self._fallback = StepTracker(
                total=self._num_steps,
                desc=self._desc,
                log_interval=max(1, self._num_steps // 10),
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._progress is not None:
            self._progress.stop()
        if self._fallback is not None:
            self._fallback.close()
        return False  # Don't suppress exceptions

    def advance(self, step: Optional[int] = None) -> None:
        """
        Advance progress by one step.

        Args:
            step: Optional explicit step number (1-indexed).
                  If not provided, auto-increments.
        """
        if self._disable:
            return

        if self._use_rich and self._progress is not None and self._task is not None:
            self._progress.advance(self._task)
            completed = int(self._progress.tasks[self._task].completed)
            self._progress.update(
                self._task,
                info=f"step {completed}/{self._num_steps}",
            )
        elif self._fallback is not None:
            self._fallback.step()

    def update(self, info: str) -> None:
        """
        Update the info text displayed in progress bar.

        Args:
            info: New info text to display.
        """
        if self._disable:
            return

        if self._use_rich and self._progress is not None and self._task is not None:
            self._progress.update(self._task, info=info)


class StageProgress:
    """
    Progress display for multi-stage operations.

    Shows which stage is currently active with visual indicators.
    Useful for showing loading/processing stages.

    Args:
        stages: List of stage names.
        desc: Overall description.

    Example:
        with StageProgress(["Text Encoder", "Transformer", "VAE"]) as progress:
            encoder = load_encoder()
            progress.advance("Loading text encoder")

            transformer = load_transformer()
            progress.advance("Loading transformer")

            vae = load_vae()
            progress.advance("Loading VAE")
    """

    def __init__(
        self,
        stages: list[str],
        desc: str = "Loading",
    ):
        self._stages = stages
        self._desc = desc
        self._current_idx = 0
        self._start_time: Optional[float] = None

        # Rich components
        self._console = None
        self._use_rich = False

    def __enter__(self) -> "StageProgress":
        self._start_time = time.time()

        try:
            from rich.console import Console

            self._console = Console()
            self._use_rich = True
            self._console.print(f"[bold blue]{self._desc}[/bold blue]")
        except ImportError:
            logger.info(f"{self._desc}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if self._use_rich and self._console is not None:
                self._console.print(
                    f"[bold green]Complete[/bold green] in {elapsed:.1f}s"
                )
            else:
                logger.info(f"Complete in {elapsed:.1f}s")
        return False

    def advance(self, message: Optional[str] = None) -> None:
        """
        Advance to the next stage.

        Args:
            message: Optional status message.
        """
        if self._current_idx >= len(self._stages):
            return

        stage = self._stages[self._current_idx]
        status = message or stage

        if self._use_rich and self._console is not None:
            # Unicode checkmark and current indicator
            for i, s in enumerate(self._stages):
                if i < self._current_idx:
                    self._console.print(f"  [green]✓[/green] {s}")
                elif i == self._current_idx:
                    self._console.print(f"  [yellow]→[/yellow] {s}: {status}")
                else:
                    self._console.print(f"  [dim]○[/dim] {s}")
        else:
            logger.info(f"  [{self._current_idx + 1}/{len(self._stages)}] {status}")

        self._current_idx += 1


def create_denoising_callback(
    progress: SamplingProgress,
):
    """
    Create a callback function for use with generate_video().

    This wraps SamplingProgress in the callback signature expected by
    the generation functions.

    Args:
        progress: SamplingProgress instance (must be within context).

    Returns:
        Callback function compatible with generate_video().

    Example:
        with SamplingProgress(num_steps=50) as progress:
            callback = create_denoising_callback(progress)
            video = generate_video(
                model, prompt_embeds, config,
                callback=callback,
            )
    """
    import torch

    def callback(step: int, total: int, latents: torch.Tensor) -> None:
        progress.advance(step)

    return callback
