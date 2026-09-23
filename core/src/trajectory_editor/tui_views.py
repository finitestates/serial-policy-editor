"""Shared completion bridge for the persistent live terminal surfaces."""

from __future__ import annotations

class ViewLifecycle:
    """Forward a view's result to the persistent application's active request."""

    def __init__(self, *, submit) -> None:
        self.submit = submit

    def _finish(self, event, *, result=None, exception=None) -> None:
        self.submit(result=result, exception=exception)


__all__ = ["ViewLifecycle"]
