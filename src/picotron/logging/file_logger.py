"""Persistent CSV metrics and plain-text transcripts for Picotron runs."""

from __future__ import annotations

import csv
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any, Mapping

from picotron.config.config import PicotronConfig


_BASE_COLUMNS = (
    "timestamp",
    "step",
    "loss",
    "learning_rate",
    "tokens_per_second",
    "elapsed",
)


class FileLogger:
    """Write flexible metrics CSV rows and a complete timestamped run transcript."""

    def __init__(
        self,
        config: PicotronConfig | None,
        *,
        method: str,
        output_dir: str | Path | None = None,
        enabled: bool | None = None,
    ) -> None:
        if not method:
            raise ValueError("method must be non-empty.")
        self.config = config
        self.method = method
        logging_config = config.logging if config is not None else None
        configured_enabled = logging_config.file_logging if logging_config is not None else True
        self.enabled = configured_enabled if enabled is None else enabled and configured_enabled
        configured_output_dir = (
            logging_config.file_logging_output_dir if logging_config is not None else "logs"
        )
        root = Path(output_dir if output_dir is not None else configured_output_dir)
        run_name = config.general.run if config is not None else method
        self.run_directory = root / _safe_path_component(run_name)
        self.metrics_path = self.run_directory / "metrics.csv"
        self.log_path = self.run_directory / "run.log"
        self._start_time = 0.0
        self._columns: list[str] = list(_BASE_COLUMNS)
        self._file_handler: logging.FileHandler | None = None
        self._event_logger = logging.getLogger("picotron.file_logger")
        self._previous_event_log_level: int | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> "FileLogger":
        if not self.enabled:
            return self
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()
        self._load_existing_columns()
        self._attach_log_handler()
        self.log_event("INFO", f"Starting {self.method} run.")
        if self.config is None:
            self.log_event("INFO", "Startup config: no Picotron config was supplied.")
        else:
            self.log_event("INFO", "Startup config:\n" + pformat(asdict(self.config), sort_dicts=True))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if not self.enabled:
            return None
        if exc_value is not None:
            logging.getLogger("picotron.file_logger").exception(
                "%s run failed: %s", self.method, exc_value
            )
        else:
            self.log_event("INFO", f"Finished {self.method} run.")
        self._detach_log_handler()
        return None

    def log_step(
        self,
        *,
        step: int,
        loss: float,
        learning_rate: float,
        tokens_seen: int,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        """Persist one optimizer step, expanding the CSV schema for new metrics."""

        if not self.enabled:
            return
        if step <= 0:
            raise ValueError("step must be positive.")
        elapsed = time.perf_counter() - self._start_time
        tokens_per_second = tokens_seen / elapsed if elapsed > 0 else 0.0
        row: dict[str, float | int | str] = {
            "timestamp": _timestamp(),
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
            "tokens_per_second": tokens_per_second,
            "elapsed": elapsed,
        }
        row.update({name: float(value) for name, value in (metrics or {}).items()})
        with self._lock:
            self._ensure_columns(row)
            with self.metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
                writer = csv.DictWriter(metrics_file, fieldnames=self._columns, extrasaction="ignore")
                if metrics_file.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
        extra = " ".join(f"{name}={value:.6f}" for name, value in (metrics or {}).items())
        self.log_event(
            "INFO",
            f"step={step} loss={loss:.6f} learning_rate={learning_rate:.6g} "
            f"tokens_per_second={tokens_per_second:.2f} elapsed={elapsed:.2f}s {extra}".rstrip(),
        )

    def log_event(self, level: str, message: str) -> None:
        """Write an explicitly supplied event into the run transcript."""

        if self.enabled:
            getattr(self._event_logger, level.lower())(message)

    def _load_existing_columns(self) -> None:
        if not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0:
            return
        with self.metrics_path.open(newline="", encoding="utf-8") as metrics_file:
            reader = csv.DictReader(metrics_file)
            if reader.fieldnames:
                self._columns = list(reader.fieldnames)

    def _ensure_columns(self, row: Mapping[str, object]) -> None:
        new_columns = [name for name in row if name not in self._columns]
        if not new_columns:
            return
        self._columns.extend(new_columns)
        if not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0:
            return
        with self.metrics_path.open(newline="", encoding="utf-8") as metrics_file:
            existing_rows = list(csv.DictReader(metrics_file))
        with self.metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=self._columns)
            writer.writeheader()
            writer.writerows(existing_rows)

    def _attach_log_handler(self) -> None:
        self._file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(self._file_handler)
        self._previous_event_log_level = self._event_logger.level
        self._event_logger.setLevel(logging.INFO)
        logging.captureWarnings(True)

    def _detach_log_handler(self) -> None:
        if self._file_handler is not None:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        if self._previous_event_log_level is not None:
            self._event_logger.setLevel(self._previous_event_log_level)
            self._previous_event_log_level = None


def _safe_path_component(value: str) -> str:
    component = Path(value).name.strip()
    if not component or component in {".", ".."}:
        raise ValueError("general.run must resolve to a safe non-empty directory name.")
    return component


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
