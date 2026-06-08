"""Structured logging for RepoPilot.

General logs go to stderr (JSON).  The tracer uses a separate stdout handler
so its JSONL output stream remains clean and parseable by downstream tools.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging.

    * Root ``repopilot`` logger -> stderr as JSON (one-per-line).
    * ``repopilot.tracer`` logger -> stdout, plain message (JSONL passthrough).

    Does NOT install a stdout handler for the tracer — the tracer's
    ``logger.info()`` calls propagate to root where they can be captured by
    pytest's caplog fixture.  For production JSONL streaming, use shell
    redirection of stderr.

    Parameters
    ----------
    level : int
        Log level for the root logger (default: ``INFO``).
    """
    # -- root logger (stderr) --------------------------------------------------
    root = logging.getLogger("repopilot")
    if root.handlers:  # already configured (e.g. by pytest caplog)
        return root
    root.setLevel(level)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        logging.Formatter(
            '{"ts":"%(asctime)s","level":"%(levelname)s",'
            '"name":"%(name)s","msg":%(message)s}'
        )
    )
    root.addHandler(stderr_handler)

    # -- tracer: just ensure propagation to root (no dedicated stdout handler) --
    tracer = logging.getLogger("repopilot.tracer")
    tracer.setLevel(level)
    tracer.propagate = True

    return root
