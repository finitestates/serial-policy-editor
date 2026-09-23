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
            show_model_probabilities=bool(
                getattr(args, "show_model_probabilities", False)
            ),
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
        seamless=io.capabilities.seamless_review,
    )


def ephemeral_policy(args: argparse.Namespace, io: Any) -> InteractivePolicy:
    """Build the policy for a persistence-free live session."""

    return _policy(
        args,
        io,
        # LiveSession owns root-relative review semantics without a recorder.
        seamless=io.capabilities.seamless_review,
    )


__all__ = ["durable_policy", "ephemeral_policy"]
