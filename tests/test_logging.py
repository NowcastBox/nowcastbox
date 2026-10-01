from __future__ import annotations

import logging

import pytest

from nowcastbox._logging import LOGGER_NAME, get_logger, set_log_level


@pytest.fixture(autouse=True)
def _restore_logger():
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level = list(logger.handlers), logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_get_logger_hierarchy():
    assert get_logger().name == LOGGER_NAME
    assert get_logger(LOGGER_NAME).name == LOGGER_NAME
    assert get_logger("nowcastbox.core.data").name == "nowcastbox.core.data"
    assert get_logger("foo").name == "nowcastbox.foo"


def test_package_logger_has_null_handler():
    assert any(isinstance(h, logging.NullHandler) for h in get_logger().handlers)


def test_set_log_level_string_and_handler():
    set_log_level("debug")
    logger = get_logger()
    assert logger.level == logging.DEBUG
    streams = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(streams) == 1
    set_log_level(logging.INFO)  # does not add a second handler
    assert len([h for h in logger.handlers if type(h) is logging.StreamHandler]) == 1


def test_set_log_level_without_handler():
    set_log_level(logging.WARNING, add_stream_handler=False)
    assert not [h for h in get_logger().handlers if type(h) is logging.StreamHandler]


def test_set_log_level_invalid():
    with pytest.raises(ValueError, match="Unknown logging level"):
        set_log_level("loud")
