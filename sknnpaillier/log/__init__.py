"""Provide JSON logging with optional Rich output and rotating log files."""

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
    """Discard calls to the four supported logging methods.

    Returned by get_logger when logging is disabled. It performs no
    logging I/O and is not a complete logging.Logger replacement.
    """

    def debug(self, *args, **kargs):
        """Discard a logging call.

        Args:
            *args (object): Ignored positional arguments.
            **kargs (object): Ignored keyword arguments.

        Returns:
            None: No record is emitted.
        """
        return
    def info(self, *args, **kargs):
        """Discard a logging call.

        Args:
            *args (object): Ignored positional arguments.
            **kargs (object): Ignored keyword arguments.

        Returns:
            None: No record is emitted.
        """
        return
    def warning(self, *args, **kargs):
        """Discard a logging call.

        Args:
            *args (object): Ignored positional arguments.
            **kargs (object): Ignored keyword arguments.

        Returns:
            None: No record is emitted.
        """
        return
    def error(self, *args, **kargs):
        """Discard a logging call.

        Args:
            *args (object): Ignored positional arguments.
            **kargs (object): Ignored keyword arguments.

        Returns:
            None: No record is emitted.
        """
        return


class _DictToJsonFormatter(logging.Formatter):
    """Render dictionary messages for the Rich console handler.

    Attributes:
        indent (int | None): JSON indentation width.
    """

    def __init__(self, indent: int | None = None):
        """Configure JSON indentation.

        Args:
            indent (int | None): Spaces per nesting level; None uses compact JSON.

        Returns:
            None: Stores the indentation setting.
        """
        super().__init__()
        self.indent = indent

    def format(self, record):
        """Format one log record.

        Non-JSON values are converted with str. This formatter replaces dictionary
        messages on the supplied record with JSON strings.

        Args:
            record (logging.LogRecord): Record to format.

        Returns:
            str: Formatted log text.

        Raises:
            TypeError: If the record contains unsupported JSON mapping keys.
            ValueError: If JSON serialization detects a circular structure.
        """
        if isinstance(record.msg, dict):
            record.msg = json.dumps(record.msg, indent=self.indent, default=str)
            record.args = ()
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """Serialize log records as JSON objects.

    Attributes:
        indent (int | None): JSON indentation width.
    """

    def __init__(self, indent: int | None = None):
        """Configure JSON indentation.

        Args:
            indent (int | None): Spaces per nesting level; None uses compact JSON.

        Returns:
            None: Stores the indentation setting.
        """
        super().__init__()
        self.indent = indent

    def format(self, record):
        """Format one log record.

        Non-JSON values are converted with str. Dictionary messages contribute top-level
        fields and can replace the formatter metadata fields.

        Args:
            record (logging.LogRecord): Record to format.

        Returns:
            str: Formatted log text.

        Raises:
            TypeError: If the record contains unsupported JSON mapping keys.
            ValueError: If JSON serialization detects a circular structure.
        """
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
    """Create a JSON logger with console output and optional rotating files.

    Configuration parameters set to None use environment defaults.
    Direct construction with disabled=True adds a NullHandler and creates
    no files. get_logger instead returns DumbLogger for disabled logging.
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
        """Configure logging handlers and optional file rotation.

        SKNN_LOG_JSON_INDENT controls JSON indentation. The normal file excludes
        ERROR and higher records; the error file includes them. Parent directories
        of custom output paths are not created automatically.

        Args:
            formatter (logging.Formatter | None): File formatter; defaults to JsonFormatter.
            console_formatter (logging.Formatter | None): Console formatter; defaults to JSON.
            name (str): Logger name. Defaults to "sknn".
            log_level (int | None): Global level; SKNN_LOG_LEVEL defaults to DEBUG.
            path (str | None): Log directory; SKNN_LOG_PATH defaults to .sknn/log.
            disabled (bool | None): Disable output; SKNN_LOG_DISABLED defaults to false.
            console_handler_level (int | None): Console threshold; direct default is DEBUG.
            file_handler_level (int | None): File threshold; default is INFO.
            error_log (bool | None): Enable a separate error file; default is false.
            filename (str | None): File basename, otherwise the logger name.
            output_path (str | None): Normal log filename override.
            error_output_path (str | None): Error log filename override.
            to_file (bool | None): Enable the normal file handler; default is false.
            when (str | None): Rotation unit; environment default is "m" (minutes).
            interval (int | None): Rotation interval; environment default is 10.
            use_rich (bool | None): Enable Rich output; environment default is false.

        Returns:
            None: Initializes this logger and its handlers.

        Raises:
            ImportError: If Rich is requested but not installed.
            ValueError: If the rotation interval, rotation unit, or log level is invalid.
            OSError: If a configured log directory or file cannot be opened.
        """
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
    """Create an enabled logger or a no-output replacement.

    Disabled logging is resolved before constructing handlers. Unlike direct
    Log construction, the factory defaults the console threshold to INFO
    unless an argument or SKNN_LOG_CONSOLE_LEVEL overrides it.

    Args:
        name (str): Logger name. Defaults to "sknn".
        **kwargs (object): Log constructor options; explicit values take
            precedence over environment defaults.

    Returns:
        Log | DumbLogger: A new logger instance, not a cached logger by name.

    Raises:
        ImportError: If Rich is requested but not installed.
        ValueError: If logging configuration contains invalid rotation or level values.
        OSError: If configured directories or files cannot be opened.
        TypeError: If an unsupported constructor keyword is provided.
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
