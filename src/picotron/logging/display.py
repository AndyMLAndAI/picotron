"""Rich training display with a safe non-TTY fallback."""

from __future__ import annotations

import time
from typing import Any

try:
    from rich.console import Console
    from rich.console import Group
    from rich.live import Live
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when Rich is not installed.
    Console = Any  # type: ignore[assignment,misc]
    Group = None  # type: ignore[assignment]
    Live = None  # type: ignore[assignment]
    Progress = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False

try:
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is declared in requirements.
    tqdm = None  # type: ignore[assignment]
    _TQDM_AVAILABLE = False

from picotron.config.config import PicotronConfig


class TrainingDisplay:
    """Display training progress live in TTYs and through a tqdm fallback."""

    def __init__(
        self,
        config: PicotronConfig,
        *,
        total_steps: int | None = None,
        console: Console | None = None,
        plain_interval: int = 10,
    ) -> None:
        if plain_interval <= 0:
            raise ValueError("plain_interval must be positive.")
        self.config = config
        self.total_steps = total_steps
        self.console = console if console is not None else Console() if _RICH_AVAILABLE else None
        self.plain_interval = plain_interval
        self._live = None
        self._progress = None
        self._progress_task = None
        self._fallback_progress = None
        self._fallback_step = 0
        self._start_time = 0.0
        self._last_values: dict[str, float | int] = {}

    @property
    def use_live(self) -> bool:
        """Whether this display will use Rich Live output."""

        return bool(_RICH_AVAILABLE and self.console is not None and self.console.is_terminal)

    def __enter__(self) -> "TrainingDisplay":
        self._start_time = time.perf_counter()
        self._print_startup_banner()
        if self.use_live:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )
            self._progress_task = self._progress.add_task(
                "steps", total=self.total_steps
            )
            self._live = Live(
                Group(self._progress, self._render_table()),
                console=self.console,
                refresh_per_second=4,
            )
            self._live.start()
        elif _TQDM_AVAILABLE:
            self._fallback_progress = tqdm(
                total=self.total_steps,
                desc="training",
                unit="step",
            )
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._fallback_progress is not None:
            self._fallback_progress.close()
            self._fallback_progress = None

    def update(
        self,
        *,
        step: int,
        loss: float,
        learning_rate: float,
        tokens_seen: int,
    ) -> None:
        """Record and render one optimizer step."""

        self._last_values = {
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "tokens_seen": tokens_seen,
        }
        if self._live is not None:
            self._progress.update(self._progress_task, completed=step)
            if self._should_render_metrics(step):
                self._live.update(Group(self._progress, self._render_table()))
        elif self._fallback_progress is not None:
            self._fallback_progress.update(step - self._fallback_step)
            if self._should_render_metrics(step):
                self._fallback_progress.set_postfix(loss=f"{loss:.4f}", refresh=False)
            self._fallback_step = step

    def _print_startup_banner(self) -> None:
        model_config = self.config.model.model_config
        tokens_config = self.config.tokens
        learning_rate = self.config.optimizer.learning_rate_scheduler.learning_rate
        if _RICH_AVAILABLE and self.console is not None:
            banner = Table(title="Picotron training")
            banner.add_column("Setting")
            banner.add_column("Value")
            for name, value in (
                (
                    "model",
                    f"{model_config.num_hidden_layers} layers / {model_config.hidden_size} hidden",
                ),
                ("sequence length", tokens_config.sequence_length),
                ("batch size", tokens_config.micro_batch_size),
                ("learning rate", learning_rate),
            ):
                banner.add_row(name, str(value))
            self.console.print(banner)
        else:
            print(
                "Picotron training: "
                f"layers={model_config.num_hidden_layers} hidden={model_config.hidden_size} "
                f"seq_len={tokens_config.sequence_length} "
                f"batch_size={tokens_config.micro_batch_size} lr={learning_rate}"
            )

    def _render_table(self) -> Table:
        elapsed = time.perf_counter() - self._start_time
        step = int(self._last_values.get("step", 0))
        loss = float(self._last_values.get("loss", 0.0))
        learning_rate = float(
            self._last_values.get(
                "learning_rate",
                self.config.optimizer.learning_rate_scheduler.learning_rate,
            )
        )
        tokens_seen = int(self._last_values.get("tokens_seen", 0))
        tokens_per_second = tokens_seen / elapsed if elapsed > 0 else 0.0
        eta = "?"
        if self.total_steps is not None and step > 0 and tokens_per_second > 0:
            remaining_tokens = (self.total_steps - step) * tokens_seen / step
            eta = _format_duration(remaining_tokens / tokens_per_second)

        table = Table(title="Training progress")
        for column in ("step", "loss", "learning_rate", "tokens/sec", "elapsed", "ETA"):
            table.add_column(column)
        table.add_row(
            str(step),
            f"{loss:.6f}",
            f"{learning_rate:.6g}",
            f"{tokens_per_second:.1f}",
            _format_duration(elapsed),
            eta,
        )
        return table

    def _should_render_metrics(self, step: int) -> bool:
        """Keep progress current while honoring the configured metric cadence."""

        return step % self.plain_interval == 0 or step == self.total_steps


def _format_duration(seconds: float) -> str:
    """Format a duration compactly for the progress table."""

    seconds = max(0.0, seconds)
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{remainder:02d}"
