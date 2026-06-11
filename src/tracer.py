import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger("repopilot.tracer")


class Tracer:
    def __init__(self):
        self.trace_id = uuid4().hex[:12]
        self.steps: list[dict] = []

    def log(self, step: str, input: dict, output: dict, error: str | None = None) -> None:
        """Emit a JSONL log entry to stdout and append to internal step list."""
        entry = {
            "trace_id": self.trace_id,
            "step": step,
            "ts": datetime.now(timezone.utc).isoformat(),
            "input": input,
            "output": output,
        }
        if error is not None:
            entry["error"] = error

        self.steps.append(entry)
        logger.info(json.dumps(entry))
