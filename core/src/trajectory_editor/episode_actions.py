"""Compatibility imports for the historical action module name.

New code should import actions from :mod:`trajectory_editor.core.actions`.
This module remains so saved workflows and existing callers do not need a
flag-day migration.
"""

from .core.actions import (
    PHRASE_DEFAULT_MAX_SHIFT,
    PHRASE_DEFAULT_MAX_TOKENS,
    Accept,
    EndGeneration,
    Hold,
    Phrase,
    PolicyAction,
    SelectRawRank,
    Write,
    action_from_dict,
)
from .core.errors import EditorError

__all__ = [
    "Accept",
    "EditorError",
    "EndGeneration",
    "Hold",
    "Phrase",
    "PHRASE_DEFAULT_MAX_SHIFT",
    "PHRASE_DEFAULT_MAX_TOKENS",
    "PolicyAction",
    "SelectRawRank",
    "Write",
    "action_from_dict",
]
