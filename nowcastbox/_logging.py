"""Package-wide logging helpers.

All modules obtain their logger through :func:`get_logger`, so that every logger is a
child of the ``"nowcastbox"`` logger. A :class:`logging.NullHandler` is attached to the
root package logger (library best practice): nothing is printed unless the user
configures logging or calls :func:`set_log_level`.
"""

from __future__ import annotations

import logging

__all__ = ["LOGGER_NAME", "get_logger", "set_log_level"]

LOGGER_NAME = "nowcastbox"

_package_logger = logging.getLogger(LOGGER_NAME)
_package_logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger in the ``nowcastbox`` hierarchy.

    Parameters
    ----------
    name : str, optional
        Module name, typically ``__name__``. Names that do not start with
        ``"nowcastbox"`` are prefixed with it. ``None`` returns the package logger.

    Returns
    -------
    logging.Logger
        The requested logger.

    Examples
    --------
    >>> from nowcastbox._logging import get_logger
    >>> get_logger("nowcastbox.models.em").name
    'nowcastbox.models.em'
    >>> get_logger("custom").name
    'nowcastbox.custom'
    """
    if name is None or name == LOGGER_NAME:
        return _package_logger
    if not name.startswith(LOGGER_NAME + "."):
        name = f"{LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def set_log_level(level: int | str, *, add_stream_handler: bool = True) -> None:
    """Set the verbosity of the package logger.

    Parameters
    ----------
    level : int or str
        Logging level, e.g. ``logging.INFO`` or ``"DEBUG"``.
    add_stream_handler : bool, default True
        Attach a :class:`logging.StreamHandler` to the package logger if it has none,
        so that messages are actually shown.

    Raises
    ------
    ValueError
        If ``level`` is a string that is not a valid logging level name.

    Examples
    --------
    >>> import logging
    >>> from nowcastbox._logging import set_log_level, get_logger
    >>> set_log_level("WARNING", add_stream_handler=False)
    >>> get_logger().level == logging.WARNING
    True
    """
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError(f"Unknown logging level {level!r}.")
        level = resolved
    _package_logger.setLevel(level)
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in _package_logger.handlers
    )
    if add_stream_handler and not has_stream:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        _package_logger.addHandler(handler)
