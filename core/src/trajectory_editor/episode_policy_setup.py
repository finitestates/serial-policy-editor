"""Build interactive teacher policies from the current launch configuration."""

from __future__ import annotations

import argparse
from typing import Any

from .episode_ui import InteractivePolicy, PolicyViewPreferences


def _preferences(args: argparse.Namespace) -> PolicyViewPreferences:
    preferences = getattr(args, "_policy_view_preferences", None)
    if preferences is None:
        preferences = PolicyViewPreferences(
            show=args.show_policy_rank,
            logit_view=args.logit_view,
        )
        args._policy_view_preferences = preferences
    return preferences


def _policy(
    args: argparse.Namespace,
    io: Any,
    *,
    store: Any | None = None,
    episode_id: str | None = None,
    seamless: bool,
) -> InteractivePolicy:
    return InteractivePolicy(
        io=io,
        menu_size=args.table_depth,
        search_radius=args.search_radius,
        default_hold_tokens=args.hold_default,
        phrase_max_tokens=getattr(args, "phrase_max_tokens", 16),
        phrase_max_shift=getattr(args, "phrase_max_shift", 6.0),
        context_characters=args.context_chars,
        manual_acceptance=args.manual_acceptance,
        view_preferences=_preferences(args),
        store=store,
        episode_id=episode_id,
        seamless=seamless,
    )


def durable_policy(
    args: argparse.Namespace,
    store: Any,
    episode_id: str,
    io: Any,
) -> InteractivePolicy:
    """Build the policy wired to one durable episode recorder."""

    return _policy(
        args,
        io,
        store=store,
        episode_id=episode_id,
        seamless=io.supports_live_choices,
    )


def ephemeral_policy(args: argparse.Namespace, io: Any) -> InteractivePolicy:
    """Build the policy for a persistence-free live session."""

    return _policy(
        args,
        io,
        # LiveSession owns the same root-relative interaction semantics without
        # a durable recorder; live choices therefore remain seamless.
        seamless=bool(getattr(io, "supports_live_choices", False)),
    )


__all__ = ["durable_policy", "ephemeral_policy"]
