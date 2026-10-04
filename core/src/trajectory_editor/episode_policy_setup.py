"""Build interactive teacher policies from the current launch configuration."""

from __future__ import annotations

import argparse
from typing import Any

from .episode_ui import InteractivePolicy, PolicyViewPreferences


def _preferences(args: argparse.Namespace) -> PolicyViewPreferences:
    preferences = getattr(args, "_policy_view_preferences", None)
    if preferences is None:
        overlays: set[str] = set()
        if args.logit_view in {"raw", "both"}:
            overlays.add("logit")
        if args.logit_view in {"gap", "both"}:
            overlays.add("diff")
        if getattr(args, "show_model_probabilities", False):
            overlays.add("probability")
        preferences = PolicyViewPreferences(
            show=args.show_policy_rank, overlays=frozenset(overlays),
        )
        args._policy_view_preferences = preferences
    return preferences


def _policy(
    args: argparse.Namespace,
    io: Any,
    *,
    session: Any | None = None,
    seamless: bool,
) -> InteractivePolicy:
    return InteractivePolicy(
        io=io,
        menu_size=args.table_depth,
        default_hold_tokens=args.hold_default,
        phrase_max_tokens=getattr(args, "phrase_max_tokens", 16),
        phrase_max_shift=getattr(args, "phrase_max_shift", 6.0),
        context_characters=args.context_chars,
        manual_acceptance=args.manual_acceptance,
        view_preferences=_preferences(args),
        session=session,
        seamless=seamless,
    )


def interactive_policy(
    args: argparse.Namespace,
    io: Any,
    *,
    session: Any | None = None,
) -> InteractivePolicy:
    """Build the policy against the active in-memory session."""

    return _policy(
        args,
        io,
        session=session,
        seamless=io.capabilities.seamless_review,
    )


__all__ = ["interactive_policy"]
