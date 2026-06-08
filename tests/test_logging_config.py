"""Tests for structured logging configuration."""

import logging

import pytest

from src.logging_config import setup_logging


def test_setup_logging_returns_root_logger():
    """setup_logging should return the 'repopilot' root logger."""
    logger = setup_logging()
    assert logger.name == "repopilot"


def test_setup_logging_sets_info_level():
    """Default log level should be INFO."""
    logger = setup_logging(level=logging.INFO)
    assert logger.level == logging.INFO


def test_setup_logging_sets_debug_level():
    """Custom log level is respected."""
    # Clear handlers to force fresh setup
    root = logging.getLogger("repopilot")
    root.handlers.clear()

    logger = setup_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_setup_logging_idempotent():
    """Calling setup_logging twice should not add duplicate handlers."""
    h1 = setup_logging()
    initial_handlers = len(h1.handlers)

    h2 = setup_logging()
    assert h2 is h1  # same logger
    assert len(h2.handlers) == initial_handlers


def test_tracer_logger_propagates_to_root():
    """Tracer logger should have propagate=True."""
    setup_logging()
    tracer = logging.getLogger("repopilot.tracer")
    assert tracer.propagate is True


def test_root_logger_has_stderr_handler():
    """Root logger should have a StreamHandler writing to stderr."""
    # Force fresh setup by clearing handlers
    root = logging.getLogger("repopilot")
    root.handlers.clear()

    logger = setup_logging()

    assert len(logger.handlers) >= 1
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)


def test_setup_logging_connects_child_loggers():
    """Child loggers should exist and be accessible after setup."""
    setup_logging()

    tracer = logging.getLogger("repopilot.tracer")
    http = logging.getLogger("repopilot.http_client")

    # Both should exist and have the root as parent
    assert tracer.parent.name == "repopilot"
    assert http.parent.name == "repopilot"


def test_setup_logging_cleans_up_during_teardown():
    """Setup should work after handlers are cleared (simulating teardown)."""
    root = logging.getLogger("repopilot")
    root.handlers.clear()

    logger = setup_logging()
    assert len(logger.handlers) >= 1
