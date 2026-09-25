"""A small executable model of coordinate-addressed token editing.

For a fixed (state, policy, world), observation is deterministic. Repeatedly
accepting proposals reveals an *immutable continuation*: the token trajectory
is fixed even though each successive prefix has a different distribution.
There is no advancing random generator, cache, or backend evaluation state here.

Numerically equivalent logits and a deterministic backend evaluation path are
assumed. Tiny changes near truncation thresholds, ties, or CDF boundaries can
change a deterministic trajectory; this is sensitivity to numerical inputs,
not consumption of randomness. Python floats are IEEE float64 on supported
platforms; this module does not promise bitwise cross-platform model parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import re
from typing import Protocol, Sequence, TypeAlias


RNG_SCHEME = "blake2b64-token-prefix-quantile-v2"
MIN_SEED, MAX_SEED = -(1 << 63), (1 << 63) - 1


def _tokens(ids: Sequence[int]) -> tuple[int, ...]:
    result = tuple(ids)
    if any(type(i) is not int or not 0 <= i < (1 << 63) for i in result):
        raise ValueError("token IDs must be nonnegative signed-64-bit integers")
    return result


def token_prefix_sha256(ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in _tokens(ids):
        digest.update(token_id.to_bytes(8, "little", signed=True))
    return digest.hexdigest()


@dataclass(frozen=True)
class World:
    """The stochastic identity and the offset of visible boundary zero."""

    seed: int
    stream_fingerprint: str
    coordinate_offset: int = 0

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not MIN_SEED <= self.seed <= MAX_SEED:
            raise ValueError("seed must be a signed-64-bit integer")
        if not isinstance(self.stream_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.stream_fingerprint
        ):
            raise ValueError("stream fingerprint must be a lowercase SHA-256 digest")
        if type(self.coordinate_offset) is not int or self.coordinate_offset < 0:
            raise ValueError("coordinate offset must be a nonnegative integer")

    @classmethod
    def for_prefix(cls, seed: int, initial_token_ids: Sequence[int], offset: int = 0) -> World:
        return cls(seed, token_prefix_sha256(initial_token_ids), offset)


def _uniform(world: World, coordinate: int, token_id: int | None = None) -> float:
    if type(coordinate) is not int or coordinate < 0:
        raise ValueError("sampling coordinate must be a nonnegative integer")
    if token_id is not None and (type(token_id) is not int or token_id < 0):
        raise ValueError("token ID must be a nonnegative integer")
    prefix = f"{RNG_SCHEME}:"
    if token_id is not None:
        prefix += "gumbel-max:"
    payload = f"{prefix}{world.seed}:{world.stream_fingerprint}:{coordinate}"
    if token_id is not None:
        payload += f":{token_id}"
    value = int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")
    return (value + 0.5) / float(1 << 64)


def position_uniform(world: World, coordinate: int) -> float:
    return _uniform(world, coordinate)


def position_uniform_token(world: World, coordinate: int, token_id: int) -> float:
    return _uniform(world, coordinate, token_id)


class Backend(Protocol):
    def logits(self, prefix_token_ids: Sequence[int]) -> Sequence[float]: ...
    def is_eog(self, token_id: int) -> bool: ...


@dataclass(frozen=True)
class Policy:
    """A small, replayable score and support transformation, separate from World."""

    temperature: float = 1.0
    top_k: int | None = None
    excluded_token_ids: tuple[int, ...] = ()
    biases: tuple[tuple[int, float], ...] = ()
    draw_kernel: str = "categorical"

    def __post_init__(self) -> None:
        if not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        if self.draw_kernel not in ("categorical", "gumbel-max"):
            raise ValueError("unsupported draw kernel")
        excluded = _tokens(self.excluded_token_ids)
        biases = tuple((i, float(v)) for i, v in self.biases)
        if any(type(i) is not int or i < 0 or not math.isfinite(v) for i, v in biases):
            raise ValueError("biases must contain nonnegative token IDs and finite values")
        if len(set(excluded)) != len(excluded) or len({i for i, _ in biases}) != len(biases):
            raise ValueError("duplicate token ID in policy")
        object.__setattr__(self, "excluded_token_ids", excluded)
        object.__setattr__(self, "biases", biases)

    def distribution(self, logits: Sequence[float], prefix_token_ids: Sequence[int]) -> Distribution:
        """Score candidates in float64, then sort by score and token ID.

        The prefix is available to a policy; this minimal policy uses fixed
        biases only. A client can compute history-aware logits in its backend.
        """
        values = tuple(float(v) for v in logits)
        if not values or any(not math.isfinite(v) for v in values):
            raise ValueError("logits must be a finite nonempty vector")
        _tokens(prefix_token_ids)
        if any(i >= len(values) for i in self.excluded_token_ids) or any(i >= len(values) for i, _ in self.biases):
            raise ValueError("policy token ID exceeds vocabulary")
        bias = dict(self.biases)
        eligible = [i for i in range(len(values)) if i not in self.excluded_token_ids]
        if not eligible:
            raise ValueError("policy excluded every candidate")
        if self.temperature == 0:
            winner = min(eligible, key=lambda i: (-(values[i] + bias.get(i, 0.0)), i))
            return Distribution((winner,), (1.0,), (values[winner] + bias.get(winner, 0.0),))
        scores = {i: (values[i] + bias.get(i, 0.0)) / float(self.temperature) for i in eligible}
        if any(not math.isfinite(v) for v in scores.values()):
            raise ValueError("policy produced non-finite scores")
        ids = tuple(sorted(eligible, key=lambda i: (-scores[i], i))[: self.top_k])
        weights = [math.exp(scores[i] - scores[ids[0]]) for i in ids]
        total = sum(weights)
        return Distribution(ids, tuple(w / total for w in weights), tuple(scores[i] for i in ids))


@dataclass(frozen=True)
class Distribution:
    ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.probabilities) or len(self.ids) != len(self.scores):
            raise ValueError("candidate arrays must have equal nonzero length")
        if len(set(self.ids)) != len(self.ids) or any(type(i) is not int or i < 0 for i in self.ids):
            raise ValueError("candidate IDs must be unique nonnegative integers")
        if any(not math.isfinite(p) or p < 0 for p in self.probabilities):
            raise ValueError("probabilities must be finite and nonnegative")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("probabilities must sum to one")
        if any(not math.isfinite(s) for s in self.scores):
            raise ValueError("scores must be finite")


def draw(distribution: Distribution, world: World, coordinate: int, kernel: str) -> int:
    if kernel == "categorical":
        u = position_uniform(world, coordinate)
        cumulative = 0.0
        for token_id, p in zip(distribution.ids, distribution.probabilities):
            cumulative += p
            if u < cumulative:  # production searchsorted(..., side="right")
                return token_id
        return distribution.ids[-1]
    if kernel == "gumbel-max":
        def rank(index: int) -> tuple[float, int]:
            token_id = distribution.ids[index]
            u = position_uniform_token(world, coordinate, token_id)
            # Rounding the 64-bit quantile to float64 can produce 1.0.
            noise = math.inf if u == 1.0 else -math.log(-math.log(u))
            return distribution.scores[index] + noise, -token_id

        return distribution.ids[max(range(len(distribution.ids)), key=rank)]
    raise ValueError("unsupported draw kernel")


@dataclass(frozen=True)
class State:
    initial_token_ids: tuple[int, ...]
    visible_token_ids: tuple[int, ...] = ()
    terminal_token_id: int | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not _tokens(self.initial_token_ids):
            raise ValueError("initial token prefix must be nonempty")
        _tokens(self.visible_token_ids)
        if self.terminal_token_id is not None:
            _tokens((self.terminal_token_id,))
        if (self.terminal_token_id is not None) != (self.terminal_reason == "eog"):
            raise ValueError("EOG termination requires exactly one terminal token")

    @property
    def boundary(self) -> int:
        return len(self.visible_token_ids)

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.initial_token_ids + self.visible_token_ids

    @property
    def ended(self) -> bool:
        return self.terminal_reason is not None


@dataclass(frozen=True)
class Branch:
    state: State
    policy: Policy
    world: World


@dataclass(frozen=True)
class Observation:
    boundary: int
    sampling_coordinate: int
    prefix_token_ids: tuple[int, ...]
    distribution: Distribution
    proposal_token_id: int


def observe(backend: Backend, branch: Branch) -> Observation:
    if branch.state.ended:
        raise ValueError("no live boundary after termination")
    prefix = branch.state.token_ids
    distribution = branch.policy.distribution(backend.logits(prefix), prefix)
    coordinate = branch.world.coordinate_offset + branch.state.boundary
    return Observation(branch.state.boundary, coordinate, prefix, distribution,
                       draw(distribution, branch.world, coordinate, branch.policy.draw_kernel))


def rewind(branch: Branch, boundary: int) -> Branch:
    if type(boundary) is not int or not 0 <= boundary <= branch.state.boundary:
        raise ValueError("invalid visible boundary")
    state = replace(branch.state, visible_token_ids=branch.state.visible_token_ids[:boundary],
                    terminal_token_id=None, terminal_reason=None)
    return replace(branch, state=state)


def fork(branch: Branch, boundary: int | None = None) -> Branch:
    """Copy an independent branch, retaining its policy and world identity."""
    return rewind(branch, branch.state.boundary if boundary is None else boundary)


def reroll(branch: Branch, new_seed: int) -> Branch:
    """Replace the exact seed, leaving prefix, policy, and coordinate intact."""
    return replace(branch, world=replace(branch.world, seed=new_seed))


@dataclass(frozen=True)
class Accept:
    pass


@dataclass(frozen=True)
class SelectToken:
    token_id: int


@dataclass(frozen=True)
class WriteTokens:
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Hold:
    limit: int
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError("hold limit must be positive")
        _tokens(self.stop_token_ids)


@dataclass(frozen=True)
class End:
    pass


@dataclass(frozen=True)
class SetPolicy:
    policy: Policy


@dataclass(frozen=True)
class Reroll:
    seed: int  # resolved integer; never an instruction to obtain entropy


Event: TypeAlias = Accept | SelectToken | WriteTokens | Hold | End | SetPolicy | Reroll


@dataclass(frozen=True)
class ActionResult:
    resolved_token_ids: tuple[int, ...]
    visible_token_ids: tuple[int, ...]
    stop_reason: str


def apply(backend: Backend, branch: Branch, action: Event) -> tuple[Branch, ActionResult]:
    """Resolve an intervention and return a new branch; observation is read-only."""
    if branch.state.ended:
        raise ValueError("cannot act on a terminated branch")
    if isinstance(action, SetPolicy):
        return replace(branch, policy=action.policy), ActionResult((), (), "policy")
    if isinstance(action, Reroll):
        return reroll(branch, action.seed), ActionResult((), (), "reroll")
    if isinstance(action, End):
        return replace(branch, state=replace(branch.state, terminal_reason="end")), ActionResult((), (), "end")

    resolved: list[int] = []
    visible: list[int] = []
    stop = "completed"
    if isinstance(action, Accept):
        planned = (observe(backend, branch).proposal_token_id,)
    elif isinstance(action, SelectToken):
        planned = (action.token_id,)
    elif isinstance(action, WriteTokens):
        planned = tuple(action.token_ids)
        if not planned:
            raise ValueError("write needs at least one token")
    elif isinstance(action, Hold):
        planned = ()  # each proposal depends on the previous commit
    else:
        raise TypeError("unsupported intervention")

    def commit(current: Branch, token_id: int) -> Branch:
        # Teacher-selected tokens may lie outside the policy support, but not
        # outside the backend vocabulary. Observe at every commit boundary.
        vocabulary = len(backend.logits(current.state.token_ids))
        if type(token_id) is not int or not 0 <= token_id < vocabulary:
            raise ValueError("token ID is outside the vocabulary")
        resolved.append(token_id)
        if backend.is_eog(token_id):
            return replace(current, state=replace(current.state, terminal_token_id=token_id,
                                                  terminal_reason="eog"))
        visible.append(token_id)
        return replace(current, state=replace(current.state,
                                              visible_token_ids=current.state.visible_token_ids + (token_id,)))

    if isinstance(action, Hold):
        for _ in range(action.limit):
            branch = commit(branch, observe(backend, branch).proposal_token_id)
            if branch.state.ended:
                stop = "eog"
                break
            if visible[-1] in action.stop_token_ids:
                stop = "stop-token"
                break
        else:
            stop = "requested-length"
    else:
        for token_id in planned:
            branch = commit(branch, token_id)
            if branch.state.ended:
                stop = "eog"
                break
    return branch, ActionResult(tuple(resolved), tuple(visible), stop)


@dataclass(frozen=True)
class Tape:
    """Initial situation and exact intervention history, including reroll seeds."""

    initial_state: State
    policy: Policy
    world: World
    events: tuple[Event, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        events = []
        for event in self.events:
            record = {"kind": type(event).__name__, **asdict(event)}
            events.append(record)
        return {"initial_state": asdict(self.initial_state), "policy": asdict(self.policy),
                "world": asdict(self.world), "events": events}

    @classmethod
    def from_dict(cls, record: dict) -> Tape:
        constructors = {action.__name__: action for action in
                        (Accept, SelectToken, WriteTokens, Hold, End, SetPolicy, Reroll)}

        def policy(data: dict) -> Policy:
            return Policy(**{**data, "excluded_token_ids": tuple(data["excluded_token_ids"]),
                             "biases": tuple(tuple(pair) for pair in data["biases"])})

        events = []
        for raw in record["events"]:
            kind = raw["kind"]
            data = {key: value for key, value in raw.items() if key != "kind"}
            if kind == "SetPolicy":
                data["policy"] = policy(data["policy"])
            elif kind == "WriteTokens":
                data["token_ids"] = tuple(data["token_ids"])
            elif kind == "Hold":
                data["stop_token_ids"] = tuple(data["stop_token_ids"])
            events.append(constructors[kind](**data))
        state = record["initial_state"]
        return cls(State(tuple(state["initial_token_ids"]), tuple(state["visible_token_ids"]),
                         state["terminal_token_id"], state["terminal_reason"]),
                   policy(record["policy"]), World(**record["world"]), tuple(events))


def record(branch: Branch, events: Sequence[Event]) -> Tape:
    return Tape(branch.state, branch.policy, branch.world, tuple(events))


def replay(backend: Backend, tape: Tape) -> tuple[Branch, tuple[ActionResult, ...]]:
    branch = Branch(tape.initial_state, tape.policy, tape.world)
    results = []
    for event in tape.events:
        branch, result = apply(backend, branch, event)
        results.append(result)
    return branch, tuple(results)
