import csv
import json
import logging
import os
import threading
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from typing import Any

from common import config

LOG_DIR = config.LOG_DIR
LOG_MAX_BYTES = config.LOG_MAX_BYTES
LOG_BACKUP_COUNT = config.LOG_BACKUP_COUNT
current_operation: ContextVar[dict[str, Any] | None] = ContextVar("current_operation", default=None)

_FORMAT = logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logging() -> None:
    """Create logs directory and configure a minimal console handler.

    File handlers are added per-logger by get_logger().
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # PDF parser warnings can contain bytes from the input document.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    logging.getLogger("PyPDF2").setLevel(logging.ERROR)

    # ensure there is a single console handler with WARNING level
    console = None
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler):
            console = h
            break
    if console is None:
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        console.setFormatter(_FORMAT)
        root.addHandler(console)
    else:
        console.setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to logs/{name}.log (file) and respects the root console handler.

    Multiple calls are idempotent (file handler added once per logger).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(config.LOG_LEVEL)
    logger.propagate = False

    # add a dedicated file handler once
    if not getattr(logger, "_file_handler_added", False):
        log_path = LOG_DIR / f"{name}.log"
        # ensure directory exists and file is present (touch) so handlers create it reliably
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            # create empty file if not exists
            if not log_path.exists():
                log_path.open("a", encoding="utf-8").close()
        except Exception:
            # best-effort: if touch fails, FileHandler will attempt to create file
            pass
        fh = RotatingFileHandler(str(log_path), maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
        fh.setLevel(config.LOG_LEVEL)
        fh.setFormatter(_FORMAT)
        logger.addHandler(fh)
        logger._file_handler_added = True  # type: ignore[attr-defined]
    return logger


def operation_update(**fields: Any) -> None:
    operation = current_operation.get()
    if operation is not None:
        operation.update(fields)


def log_operation(fields: dict[str, Any]) -> None:
    logger = logging.getLogger("operations")
    if not getattr(logger, "_operation_handler_added", False):
        with _operation_lock:
            if not getattr(logger, "_operation_handler_added", False):
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(LOG_DIR / "operations.jsonl", maxBytes=LOG_MAX_BYTES,
                                              backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
                logger.propagate = False
                logger._operation_handler_added = True
    logger.info(json.dumps(fields, ensure_ascii=False, separators=(",", ":")))


_operation_lock = threading.Lock()


def get_console_logger(name: str) -> logging.Logger:
    """Return a logger that only writes to the terminal, not to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not getattr(logger, "_console_handler_added", False):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(_FORMAT)
        logger.addHandler(handler)
        logger._console_handler_added = True  # type: ignore[attr-defined]
    return logger


class CsvLogger:
    def __init__(self, name: str, fieldnames: list[str] | None = None, key: str | None = None):
        self._path = LOG_DIR / f"{name}.csv"
        self._lock = threading.Lock()
        self._fieldnames: list[str] | None = fieldnames
        self._key = key

    def log(self, row: dict) -> None:
        with self._lock:
            if self._fieldnames is None:
                self._fieldnames = list(row.keys())
            else:
                for k in row:
                    if k not in self._fieldnames:
                        raise ValueError(
                            f"CsvLogger: unknown key {k!r}. "
                            f"Known fieldnames: {self._fieldnames}"
                        )
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            rotate = self._path.exists() and self._path.stat().st_size >= LOG_MAX_BYTES
            if self._path.exists() and self._path.stat().st_size and not rotate:
                with self._path.open("r", newline="", encoding="utf-8") as f:
                    rotate = next(csv.reader(f, delimiter=";"), []) != self._fieldnames
            if rotate:
                for number in range(LOG_BACKUP_COUNT, 0, -1):
                    source = self._path if number == 1 else self._path.with_suffix(f".csv.{number - 1}")
                    target = self._path.with_suffix(f".csv.{number}")
                    if source.exists():
                        os.replace(source, target)

            if self._key and self._key in row and self._path.exists() and self._path.stat().st_size > 0:
                key_val = row[self._key]
                with open(self._path, "r", newline="", encoding="utf-8") as f:
                    all_rows = list(csv.DictReader(f, delimiter=';'))

                if all_rows:
                    for i, r in enumerate(all_rows):
                        if r.get(self._key) == key_val:
                            all_rows[i] = {**r, **row}
                            with open(self._path, "w", newline="", encoding="utf-8") as f:
                                w = csv.DictWriter(f, fieldnames=self._fieldnames, delimiter=';')
                                w.writeheader()
                                w.writerows(all_rows)
                            return

            exists = self._path.exists() and self._path.stat().st_size > 0
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames, delimiter=';')
                if not exists:
                    w.writeheader()
                w.writerow(row)
