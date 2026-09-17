"""Compatibility seam for the runtime controller pipeline.

The pipeline deliberately delegates arithmetic to ``ObservationStatistics``
in this first executable slice. It provides one construction point and an
opt-in intermediate-surface trace so later controller extraction can happen
without changing replay math in the same commit.
"""

from __future__ import annotations

from typing import Any

from .sampling import ObservationStatistics


class ControllerPipeline:
    """Build policy observations through one trace-capable runtime seam."""

    def __init__(self, *, capture_trace: bool = False) -> None:
        self.capture_trace = bool(capture_trace)

    def build_statistics(
        self,
        logits: Any,
        config: Any,
        history_token_ids: Any,
        boundaries: Any = None,
        **kwargs: Any,
    ) -> ObservationStatistics:
        return ObservationStatistics(
            logits,
            config,
            history_token_ids,
            boundaries,
            capture_trace=self.capture_trace,
            **kwargs,
        )
