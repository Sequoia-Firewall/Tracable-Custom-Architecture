import os

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme

_THEME = Theme({
    "log.info":    "bright_green",
    "log.warning": "yellow",
    "log.error":   "bold red",
    "log.debug":   "cyan",
    "progress.description": "bold white",
})

_STYLES = {4: "log.info", 3: "log.warning", 2: "log.error", 1: "log.debug"}
_LABELS = {4: "INFO", 3: "WARN", 2: " ERR", 1: " DBG"}


class ProgressBar:
    """
    Thin shim that wraps a (rich.Progress, task_id) pair and exposes the
    tqdm-compatible update() / set_description() / close() API that internal
    callback functions in Main already use.  Allows existing code to keep its
    `bar` parameter without modification.
    """

    def __init__(self, progress: Progress, task_id) -> None:
        self._progress = progress
        self._task_id  = task_id

    def update(self, n: int = 1) -> None:
        self._progress.update(self._task_id, advance=n)

    def set_description(self, desc: str) -> None:
        self._progress.update(self._task_id, description=desc)

    def close(self) -> None:
        """No-op – rich Progress manages task lifetime automatically."""


class RichLogger:
    """
    Drop-in replacement for Logger that routes all terminal output through a
    shared rich Console, providing colour-coded log lines and rich progress
    bars while keeping identical file-logging behaviour.

    The .log(message, classification, Loud) signature is unchanged so every
    .display() call in Main and all components works without any modification.

    Classification levels:
        4 = INFO    (bright green)
        3 = WARNING (yellow)
        2 = ERROR   (bold red)
        1 = DEBUG   (cyan)
    """

    # Kept for any code that inspects Logger.classifications
    classifications = {
        4: "[INFO]: ", 3: "[WARNING]: ", 2: "[ERROR]: ", 1: "[DEBUG]: "
    }

    def __init__(self, filename: str, log_level: int) -> None:
        self.logs_folder    = "logs/"
        self.filename       = filename
        self.log_level      = log_level
        self.console        = Console(theme=_THEME, highlight=False)
        self.log_check_count = 10
        self._buffer        = []   # list of (message, classification, Loud)
        self._call_count    = 0
        self._preamble_counts = {}  # preamble -> total count seen
        self._preamble_log_threshold = 1000  # write a condensed log at most every N repeated messages

    # ------------------------------------------------------------------
    # Core log method – same signature as Logger.log()
    # ------------------------------------------------------------------

    def _write(self, message: str, classification: int, Loud: bool) -> None:
        """Write a single message to file and/or terminal immediately."""
        os.makedirs(self.logs_folder, exist_ok=True)
        prefix = self.classifications.get(classification, "[INFO]: ")
        if classification <= self.log_level:
            try:
                with open(self.logs_folder + self.filename, "a") as fh:
                    fh.write(f"{prefix}{message}\n")
            except Exception:
                pass
        if Loud:
            style    = _STYLES.get(classification, "log.info")
            label    = _LABELS.get(classification, "INFO")
            safe_msg = escape(message)
            self.console.print(f"[{style}]\\[[bold]{label}[/bold]][/] {safe_msg}")

    def _flush_buffer(self) -> None:
        """Flush buffered messages, condensing repeated 10-char preambles."""
        from collections import OrderedDict
        groups = OrderedDict()
        for msg, cls, loud in self._buffer:
            preamble = msg[:10]
            if preamble not in groups:
                groups[preamble] = []
            groups[preamble].append((msg, cls, loud))

        for preamble, entries in groups.items():
            first_msg, first_cls, first_loud = entries[0]
            if preamble not in self._preamble_counts:
                self._preamble_counts[preamble] = len(entries)
                self._write(first_msg, first_cls, first_loud)
            else:
                old_count = self._preamble_counts[preamble]
                new_count = old_count + len(entries)
                self._preamble_counts[preamble] = new_count
                if old_count // self._preamble_log_threshold != new_count // self._preamble_log_threshold:
                    self._write(f"{first_msg} (repeated {new_count}x total)", first_cls, first_loud)
        self._buffer.clear()

    def log(self, message: str, classification: int, Loud: bool) -> None:
        if classification <= 2:  # ERROR / DEBUG — flush and write immediately
            self._flush_buffer()
            self._call_count = 0
            self._write(message, classification, Loud)
        else:
            self._buffer.append((message, classification, Loud))
            self._call_count += 1
            if self._call_count >= self.log_check_count:
                self._flush_buffer()
                self._call_count = 0

    # ------------------------------------------------------------------
    # Progress-bar factory
    # ------------------------------------------------------------------

    def make_progress(
        self,
        transient: bool = False,
        disable:   bool = False,
    ) -> Progress:
        """
        Return a rich Progress pre-configured with the shared console theme.
        Use as a context manager wherever tqdm was previously used:

            with self.Logger.make_progress() as progress:
                task = progress.add_task("Doing work", total=100)
                for item in items:
                    ...
                    progress.update(task, advance=1)

        For callbacks that still accept a tqdm-like bar object, wrap a task
        with ProgressBar(progress, task_id).

        Parameters
        ----------
        transient : bool
            If True the progress display is erased after the context exits.
        disable : bool
            If True no progress output is rendered (for silent/quiet modes).
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", justify="left"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=transient,
            disable=disable,
        )
