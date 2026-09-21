"""Research reference-prior trie and conditional bias calculations.

These helpers are used by saved research configuration records independently
of the retired sampling observer.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class _ReferenceTrieNode:
    children: dict[int, int]
    failure: int = 0
    descendant_mass: float = 0.0
    terminal_mass: float = 0.0
    prefix: tuple[int, ...] = ()


class ReferencePriorTrie:
    """Weighted token trie with deterministic suffix/failure transitions."""

    def __init__(self, routes) -> None:
        self.nodes = [_ReferenceTrieNode(children={})]
        for route, weight in routes:
            self._insert(tuple(route), float(weight))
        self._build_failure_links()

    def _insert(self, route: tuple[int, ...], weight: float) -> None:
        node_index = 0
        self.nodes[node_index].descendant_mass += weight
        prefix: list[int] = []
        for token in route:
            prefix.append(int(token))
            child = self.nodes[node_index].children.get(int(token))
            if child is None:
                child = len(self.nodes)
                self.nodes[node_index].children[int(token)] = child
                self.nodes.append(_ReferenceTrieNode(
                    children={}, prefix=tuple(prefix)
                ))
            node_index = child
            self.nodes[node_index].descendant_mass += weight
        self.nodes[node_index].terminal_mass += weight

    def _build_failure_links(self) -> None:
        queue: deque[int] = deque()
        for child in self.nodes[0].children.values():
            self.nodes[child].failure = 0
            queue.append(child)
        while queue:
            node_index = queue.popleft()
            node = self.nodes[node_index]
            for token, child in node.children.items():
                failure = node.failure
                while failure and token not in self.nodes[failure].children:
                    failure = self.nodes[failure].failure
                self.nodes[child].failure = self.nodes[failure].children.get(token, 0)
                queue.append(child)

    def transition(self, state: int, token: int) -> int:
        token = int(token)
        while state and token not in self.nodes[state].children:
            state = self.nodes[state].failure
        return self.nodes[state].children.get(token, 0)

    def state_for_history(self, history) -> int:
        state = 0
        for token in history:
            state = self.transition(state, int(token))
        # A completed terminal route releases the state unless it is also a
        # prefix of a longer route.  Failure preserves overlapping suffixes.
        while state and not self.nodes[state].children:
            state = self.nodes[state].failure
        return state

    def outgoing(self, state: int) -> tuple[tuple[int, float], ...]:
        node = self.nodes[state]
        return tuple(
            (token, self.nodes[child].descendant_mass)
            for token, child in sorted(node.children.items())
        )

    def diagnostics(self, state: int) -> dict[str, object]:
        node = self.nodes[state]
        return {
            "state_prefix": list(node.prefix),
            "root_mass": self.nodes[0].descendant_mass,
            "state_mass": node.descendant_mass,
            "terminal_mass": node.terminal_mass,
            "outgoing": {
                token: mass for token, mass in self.outgoing(state)
            },
        }


@lru_cache(maxsize=32)
def reference_trie(routes):
    """Reuse immutable route structure when a strength or learner state changes."""
    return ReferencePriorTrie(routes)


@dataclass(frozen=True)
class ReferencePriorSnapshot:
    scope: str
    mode: str
    state_prefix: tuple[int, ...]
    root_mass: float
    state_mass: float
    terminal_mass: float
    outgoing: tuple[tuple[int, float, float, float, float, float], ...]
    biases: dict[int, float]


def reference_prior_snapshot(
    routes,
    history_token_ids,
    *,
    active_routes=None,
    strength: float,
    attraction: float,
    exit_strength: float = 0.25,
    scope: str = "global",
    mode: str = "contrastive",
    trie: ReferencePriorTrie | None = None,
) -> ReferencePriorSnapshot:
    """Evaluate one stateful lexical prior from model-visible token history."""

    # Accept the pre-mode scope names at this low-level boundary so catalogs
    # or callers created by the previous experimental interface remain easy
    # to inspect while the saved runtime representation stays normalized.
    if scope == "ballistic-global":
        scope, mode = "global", "ballistic"
    elif scope == "ballistic-global-exit":
        scope, mode = "global", "ballistic-exit"
    if scope not in {"active", "global"}:
        raise ValueError("reference prior scope must be active or global")
    if mode not in {"lexical", "contrastive", "contrastive-exit", "ballistic", "ballistic-exit"}:
        raise ValueError("unknown reference prior mode")

    selected_routes = routes
    if active_routes is not None:
        selected_routes = tuple(
            (route, weight) for route, weight in routes if route in active_routes
        )
    if not selected_routes:
        return ReferencePriorSnapshot(scope, mode, (), 0.0, 0.0, 0.0, (), {})
    if history_token_ids is None:
        if any(len(route) > 1 for route, _weight in selected_routes):
            raise ValueError("reference priors require exact context token IDs")
        history = ()
    else:
        history = tuple(int(token) for token in history_token_ids)
    if trie is None or active_routes is not None:
        trie = reference_trie(tuple(selected_routes))
    state = trie.state_for_history(history)
    node = trie.nodes[state]
    outgoing = trie.outgoing(state)
    if not outgoing:
        return ReferencePriorSnapshot(
            scope, mode, node.prefix, trie.nodes[0].descendant_mass,
            node.descendant_mass, node.terminal_mass, (), {},
        )

    log_masses = {token: math.log(mass) for token, mass in outgoing}
    center = sum(log_masses.values()) / len(log_masses)
    continuation_mass = max(0.0, node.descendant_mass - node.terminal_mass)
    exit_continue = 0.0
    if mode.endswith("-exit") and node.terminal_mass > 0.0 and continuation_mass > 0.0:
        # Terminal mass is an implicit EXIT option. Since no decoder token
        # represents EXIT, apply its log-odds against CONTINUE uniformly to
        # all continuation children. This is separate from relative child
        # scoring and from lexical commitment attraction.
        exit_continue = float(exit_strength) * math.log(
            continuation_mass / node.terminal_mass
        )

    state_attraction = 0.0
    root_attraction = mode.startswith("ballistic")
    if attraction > 0.0 and (state or root_attraction):
        # The constant term makes a singleton continuation attractive even
        # when it is the only route and therefore has no branch contrast.
        state_attraction = float(attraction) * (
            1.0 + math.log(trie.nodes[0].descendant_mass / node.descendant_mass)
        )
    rows = []
    biases = {}
    for token, mass in outgoing:
        branch = float(strength) * (log_masses[token] - center)
        total = branch + state_attraction + exit_continue
        if mode == "lexical":
            # Relative preference at every lexical branch, with modest support
            # for completing a prefix the model has already entered. Root
            # words receive no unconditional attraction. Extreme weights have
            # bounded influence and unlisted decoder tokens stay available.
            commitment = float(strength) if state and not node.terminal_mass else 0.0
            if state and node.terminal_mass and continuation_mass:
                commitment = float(strength) * math.log(continuation_mass / node.terminal_mass)
            total = max(-2.0, min(2.0, branch + commitment))
            state_attraction = commitment
        rows.append((token, mass, branch, state_attraction, exit_continue, total))
        biases[token] = total
    return ReferencePriorSnapshot(
        scope, mode, node.prefix, trie.nodes[0].descendant_mass,
        node.descendant_mass, node.terminal_mass, tuple(rows), biases,
    )


def reference_prior_biases(
    routes,
    history_token_ids,
    *,
    active_routes=None,
    strength: float,
    attraction: float = 0.0,
    exit_strength: float = 0.25,
    scope: str = "global",
    mode: str = "contrastive",
) -> dict[int, float]:
    """Compatibility wrapper returning only online prior logit adjustments."""

    return reference_prior_snapshot(
        routes,
        history_token_ids,
        active_routes=active_routes,
        strength=strength,
        attraction=attraction,
        exit_strength=exit_strength,
        scope=scope,
        mode=mode,
    ).biases

