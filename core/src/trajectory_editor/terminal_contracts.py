"""Renderer-independent terminal requests and responses.

The episode-owning thread prepares these values. A renderer may display them,
but command interpretation and engine work stay with the caller.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .core.candidates import Candidate
from .core.ui import ChoiceSet, InsertMode


# Enter in seamless review is distinct from Escape and ordinary command text.
SEAMLESS_REACTIVATE = "\x1e"


@dataclass(frozen=True)
class ChoiceFeedback:
    category: str
    title: str
    lines: tuple[str, ...] = ()
    completion_commands: tuple[str, ...] = ()
    initial_tab_command: str | None = None


@dataclass(frozen=True)
class BoundaryReview:
    active_aligned_step: int
    aligned_step: int
    context_text_tail: str
    position: Mapping[str, Any]
    next_token: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ChoiceViewState:
    """One prepared teacher decision, including optional read-only review."""

    choice: ChoiceSet
    remaining_tokens: int | None
    candidates: tuple[Candidate, ...]
    resolve_insertion: Callable[[str, InsertMode], str]
    display_candidates: tuple[Candidate, ...] | None = None
    resolve_candidate: Callable[[int], Candidate] | None = None
    target_token_id: int | None = None
    feedback: ChoiceFeedback | None = None
    initial_command: str | None = None
    review: BoundaryReview | None = None
    seamless: bool = False
    reactivate_on_review_enter: bool = False
    search_lens_active: bool = False
    policy_active: bool = False
    show_policy_rank: bool = False
    sort_by_policy: bool = False
    logit_view: str = "none"
    show_model_probabilities: bool = False
    column_focus: str | None = None
    overlays: frozenset[str] = frozenset()
    default_hold_tokens: int = 100
    default_search_radius: int = 3
    warm_selection: Callable[[int, int, int, Callable[[], bool]], Any] | None = field(
        default=None, repr=False, compare=False
    )
    cancel_warm_selection: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )
    search_warm_target: tuple[int, int] | None = None
    search_warm_commands: tuple[str, ...] = ()
    search_warm_prepared: bool = False


@dataclass(frozen=True)
class EdgeViewState:
    episode_id: str
    boundary: int
    current_budget: int | None
    remaining_tokens: int | None
    sampler_summary: str
    mode: str = "episode"


@dataclass(frozen=True)
class PromptRequest:
    """Ordinary input, a confirmation/key, composition, page, or isolated chord."""

    prompt: str
    body: str = ""
    single_key: bool = False
    page: bool = False
    multiline: bool = False
    isolated: bool = False


@dataclass(frozen=True)
class TerminalCapabilities:
    live_views: bool
    columns: int | None = None
    rows: int | None = None
    single_key: bool = False
    seamless_review: bool = False


class TerminalProtocol(Protocol):
    @property
    def capabilities(self) -> TerminalCapabilities: ...

    def terminal_size(self) -> tuple[int, int] | None: ...

    def session(self) -> AbstractContextManager[object | None]: ...

    def read_choice(self, state: ChoiceViewState) -> str | None: ...

    def read_edge(self, state: EdgeViewState) -> str | None: ...

    def prompt(self, request: PromptRequest) -> str | None: ...

    def read(self, prompt: str) -> str | None: ...

    def read_key(self, prompt: str) -> str | None: ...

    def write(self, text: str = "", *, end: str = "\n") -> None: ...

    def page(self, text: str) -> None: ...


class IO(Protocol):
    """Small text-only subset used by existing runtime and display callers."""

    def read(self, prompt: str) -> str | None: ...

    def read_key(self, prompt: str) -> str | None: ...

    def write(self, text: str = "", *, end: str = "\n") -> None: ...

    def page(self, text: str) -> None: ...
