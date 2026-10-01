"""Shared Hypothesis budgets for local exploration and repeatable CI runs."""

import os

from hypothesis import settings


# Retain Hypothesis's example database, shrinking, and failure reproduction
# output. Avoid wall-clock deadlines for state machines that exercise async UI.
settings.register_profile(
    "dev",
    max_examples=50,
    stateful_step_count=30,
    deadline=None,
    print_blob=True,
)
settings.register_profile(
    "ci",
    parent=settings.get_profile("dev"),
    derandomize=True,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
