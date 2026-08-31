"""
NotebookLogger
==============
Drop-in replacement for RichLogger for use in Jupyter notebooks.
Uses tqdm for progress bars and plain print for console output
instead of rich, which does not render correctly in notebooks.

The .log(message, classification, Loud) signature is identical to
Logger / RichLogger so all .display() calls in SegmentHandler work
without modification.

Because this class does not expose a .console attribute, all
getattr(logger, 'console', None) checks in SegmentHandler return
None, triggering their plain-text fallbacks automatically.
"""

from Components.Logger import Logger

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


class _TqdmProgress:
    """Context manager that mimics the rich.Progress API used by SegmentHandler.

    Supported methods:
        add_task(description, total)  -> task_id
        update(task_id, advance, description)
    """

    def __init__(self, transient=False, disable=False):
        self._transient = transient
        self._disable   = disable
        self._bars      = {}
        self._next_id   = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for bar in self._bars.values():
            if bar is not None:
                bar.close()
        self._bars.clear()

    def add_task(self, description='', total=100, **kwargs):
        task_id = self._next_id
        self._next_id += 1
        if _TQDM_AVAILABLE:
            bar = _tqdm(
                total=total,
                desc=description,
                disable=self._disable,
                leave=not self._transient,
            )
        else:
            bar = None
        self._bars[task_id] = bar
        return task_id

    def update(self, task_id, advance=0, description=None, **kwargs):
        bar = self._bars.get(task_id)
        if bar is None:
            return
        if description is not None:
            bar.set_description(description)
        if advance:
            bar.update(advance)


class NotebookLogger(Logger):
    """Logger for Jupyter notebooks.

    Mirrors RichLogger's _write / _flush_buffer / log architecture so that
    classification-level filtering is applied cleanly and independently of
    file I/O.  Uses plain print() for console output instead of rich so that
    output renders correctly in Jupyter cells.

    No .console attribute is exposed, so SegmentHandler's rich table
    rendering paths are bypassed and their plain-text fallbacks are used
    instead.

    Classification levels (same as Logger / RichLogger):
        4 = INFO    — verbose operational messages
        3 = WARNING — potential issues
        2 = ERROR   — errors, flushed immediately
        1 = DEBUG   — low-level detail, flushed immediately

    Filtering:
        log_level     — maximum classification written to file  (file receives
                        messages where classification <= log_level)
        console_level — minimum classification printed to stdout (console shows
                        messages where classification >= console_level)
    """

    # Expose the same classifications dict as Logger / RichLogger
    classifications = {
        4: "[INFO]: ", 3: "[WARNING]: ", 2: "[ERROR]: ", 1: "[DEBUG]: "
    }

    def __init__(self, filename: str, log_level: int, console_level: int) -> None:
        super().__init__(filename, log_level, console_level)
        self.disable_progress: bool = False  # set True to silence all progress bars

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, message: str, classification: int, Loud: bool) -> None:
        """Write one message to file and/or stdout with classification filtering."""
        if not self._logs_folder_ready:
            import os
            os.makedirs(self.logs_folder, exist_ok=True)
            self._logs_folder_ready = True
        prefix = self.classifications.get(classification, "[INFO]: ")
        if classification <= self.log_level:
            try:
                with open(self.logs_folder + self.filename, "a") as fh:
                    fh.write(f"{prefix}{message}\n")
            except Exception:
                pass
        if Loud and classification >= self.console_level:
            print(f"{prefix}{message}")

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
                    self._write(
                        f"{first_msg} (repeated {new_count}x total)",
                        first_cls, first_loud,
                    )
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Public log method — same signature as Logger / RichLogger
    # ------------------------------------------------------------------

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

    def make_progress(self, transient: bool = False, disable: bool = False) -> _TqdmProgress:
        """Return a tqdm-backed progress context manager.

        Usage mirrors rich.Progress:

            with logger.make_progress() as progress:
                task = progress.add_task("Doing work", total=n)
                for item in items:
                    ...
                    progress.update(task, advance=1)

        Set ``logger.disable_progress = True`` to suppress all bars globally.
        """
        return _TqdmProgress(transient=transient, disable=disable or self.disable_progress)
