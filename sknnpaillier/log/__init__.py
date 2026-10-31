import json
import logging
import os
import sys
import threading
from logging.handlers import TimedRotatingFileHandler

try:
    from rich.console import Console as RichConsole
    from rich.highlighter import JSONHighlighter
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class DumbLogger:
    """No-op logger that silently discards all log calls.

    Drop-in replacement for Log when logging is disabled — calling any
    method on this object is a no-op with no logging I/O.
    """

    def debug(self, *args, **kargs):
        return
    def info(self, *args, **kargs):
        return
    def warning(self, *args, **kargs):
        return
    def error(self, *args, **kargs):
        return


class _DictToJsonFormatter(logging.Formatter):
    """Converts dict log messages to indented JSON strings for RichHandler."""

    def __init__(self, indent: int | None = None):
        super().__init__()
        self.indent = indent

    def format(self, record):
        if isinstance(record.msg, dict):
            record.msg = json.dumps(record.msg, indent=self.indent, default=str)
            record.args = ()
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """Logging formatter that serialises log records as JSON objects.

    Pass ``indent=4`` (or any int) for human-readable indented output.
    Leave ``indent=None`` (default) for compact single-line NDJSON suitable
    for log files and tools like ``jq``.
    """

    def __init__(self, indent: int | None = None):
        super().__init__()
        self.indent = indent

    def format(self, record):
        thread_id = threading.current_thread().name
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger_name': record.name,
            "thread_name": thread_id,
        }
        if isinstance(record.msg, dict):
            log_data.update(record.msg)
        else:
            log_data['message'] = record.getMessage()
        return json.dumps(log_data, default=str, indent=self.indent)


class Log(logging.Logger):
    """Structured JSON logger with rotating file and console handlers.

    Extends ``logging.Logger`` to add JSON-formatted output, optional
    timed-rotating file handlers, and a dedicated error log file.  When
    ``disabled=True`` no handlers are attached and no filesystem paths
    are created, making it safe to instantiate in environments where
    disk I/O is unavailable or unwanted.

    When ``use_rich=True`` the console handler uses ``RichHandler`` with
    ``JSONHighlighter`` for syntax-coloured JSON output.  Requires
    ``pip install rich``.
    """

    def __init__(self,
                 formatter: logging.Formatter | None = None,
                 console_formatter: logging.Formatter | None = None,
                 name: str = "sknn",
                 log_level: int | None = None,
                 path: str | None = None,
                 disabled: bool | None = None,
                 console_handler_level: int | None = None,
                 file_handler_level: int | None = None,
                 error_log: bool | None = None,
                 filename: str | None = None,
                 output_path: str | None = None,
                 error_output_path: str | None = None,
                 to_file: bool | None = None,
                 when: str | None = None,
                 interval: int | None = None,
                 use_rich: bool | None = None,
    ):
        def _bool(value: str) -> bool:
            return value.lower() in ("1", "true", "yes")

        def _level(value: str) -> int:
            return getattr(logging, value.upper(), logging.DEBUG)

        if path is None:
            path = os.environ.get("SKNN_LOG_PATH", ".sknn/log")
        if disabled is None:
            disabled = _bool(os.environ.get("SKNN_LOG_DISABLED", "0"))
        if log_level is None:
            log_level = _level(os.environ.get("SKNN_LOG_LEVEL", "DEBUG"))
        if to_file is None:
            to_file = _bool(os.environ.get("SKNN_LOG_TO_FILE", "0"))
        if when is None:
            when = os.environ.get("SKNN_LOG_ROTATION_WHEN", "m")
        if interval is None:
            interval = int(os.environ.get("SKNN_LOG_ROTATION_INTERVAL", "10"))
        if use_rich is None:
            use_rich = _bool(os.environ.get("SKNN_LOG_RICH", "0"))
        if error_log is None:
            error_log = _bool(os.environ.get("SKNN_LOG_ERROR_FILE", "0"))
        if console_handler_level is None:
            console_handler_level = _level(os.environ.get("SKNN_LOG_CONSOLE_LEVEL", "DEBUG"))
        if file_handler_level is None:
            file_handler_level = _level(os.environ.get("SKNN_LOG_FILE_LEVEL", "INFO"))

        _raw    = os.environ.get("SKNN_LOG_JSON_INDENT", "0")
        _indent = int(_raw) if _raw.isdigit() and int(_raw) > 0 else None


        super().__init__(name, log_level)
        self.propagate = False
        if disabled:
            self.addHandler(logging.NullHandler())
            return
        if console_formatter is None:
            console_formatter = JsonFormatter(indent=_indent)
        if formatter is None:
            formatter = JsonFormatter()

        if to_file or error_log:
            os.makedirs(path, exist_ok=True)

        if use_rich:
            if not _RICH_AVAILABLE:
                raise ImportError(
                    "rich is not installed. Install it with: pip install rich"
                )
            console_handler = RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                highlighter=JSONHighlighter(),
                console=RichConsole(file=sys.stdout),
            )
            console_handler.setFormatter(_DictToJsonFormatter(indent=_indent))
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)

        console_handler.setLevel(console_handler_level)
        self.addHandler(console_handler)

        if to_file:
            filehandler = TimedRotatingFileHandler(
                filename=output_path or "{}/{}.log".format(path, filename or name),
                when=when,
                interval=interval,
            )
            filehandler.setFormatter(formatter)
            filehandler.setLevel(file_handler_level)
            filehandler.addFilter(lambda r: r.levelno < logging.ERROR)
            self.addHandler(filehandler)

        if error_log:
            error_file_handler = TimedRotatingFileHandler(
                filename=error_output_path or "{}/{}.error.log".format(path, filename or name),
                when=when,
                interval=interval,
            )
            error_file_handler.setFormatter(formatter)
            error_file_handler.setLevel(logging.ERROR)
            error_file_handler.addFilter(lambda record: record.levelno >= logging.ERROR)
            self.addHandler(error_file_handler)


def get_logger(name: str = "sknn", **kwargs) -> Log | DumbLogger:
    """Fábrica de loggers del proyecto.

    Devuelve ``DumbLogger`` (sin I/O de logging) cuando el logging está
    deshabilitado con ``SKNN_LOG_DISABLED=1``; en caso contrario devuelve
    un ``Log`` con salida JSON a consola en nivel INFO por defecto
    (ajustable con ``SKNN_LOG_CONSOLE_LEVEL=DEBUG``).

    El nivel global del logger sigue siendo DEBUG (o el que fije
    ``SKNN_LOG_LEVEL``); el filtro por nivel lo aplica cada handler.
    """
    disabled = kwargs.pop("disabled", None)
    if disabled is None:
        disabled = os.environ.get(
            "SKNN_LOG_DISABLED", "0"
        ).lower() in ("1", "true", "yes")
    if disabled:
        return DumbLogger()
    if (
        "console_handler_level" not in kwargs
        and "SKNN_LOG_CONSOLE_LEVEL" not in os.environ
    ):
        kwargs["console_handler_level"] = logging.INFO
    return Log(name=name, disabled=False, **kwargs)
