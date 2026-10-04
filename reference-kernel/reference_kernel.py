"""A small executable model of boundary-addressed token editing.

For a fixed (state, policy, world), observation is deterministic. Repeatedly
accepting proposals reveals an *immutable continuation*: the token trajectory
is fixed even though each successive prefix has a different distribution.
There is no advancing random generator, cache, or backend evaluation state here.

Numerically equivalent logits and a deterministic backend evaluation path are
assumed. Tiny changes near truncation thresholds, ties, or noise winner boundaries can
change a deterministic trajectory; this is sensitivity to numerical inputs,
not consumption of randomness. Python floats are IEEE float64 on supported
platforms; this module does not promise bitwise cross-platform model parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import Counter
import hashlib
import math
import re
from typing import Protocol, Sequence, TypeAlias


RNG_SCHEME = "blake2b64-token-prefix-quantile-v2"
DRAW_KERNELS = ("argmax", "gumbel-max", "gaussian-max", "logistic-max", "laplace-max", "uniform-max", "student-t-max")
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
    """The stochastic identity for one root prompt."""

    seed: int
    stream_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not MIN_SEED <= self.seed <= MAX_SEED:
            raise ValueError("seed must be a signed-64-bit integer")
        if not isinstance(self.stream_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.stream_fingerprint
        ):
            raise ValueError("stream fingerprint must be a lowercase SHA-256 digest")
    @classmethod
    def for_prefix(cls, seed: int, initial_token_ids: Sequence[int]) -> World:
        return cls(seed, token_prefix_sha256(initial_token_ids))


def _uniform(world: World, boundary: int, token_id: int | None = None) -> float:
    if type(boundary) is not int or boundary < 0:
        raise ValueError("sampling boundary must be a nonnegative integer")
    if token_id is not None and (type(token_id) is not int or token_id < 0):
        raise ValueError("token ID must be a nonnegative integer")
    prefix = f"{RNG_SCHEME}:"
    if token_id is not None:
        prefix += "gumbel-max:"
    payload = f"{prefix}{world.seed}:{world.stream_fingerprint}:{boundary}"
    if token_id is not None:
        payload += f":{token_id}"
    value = int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")
    return (value + 0.5) / float(1 << 64)


def position_uniform(world: World, boundary: int) -> float:
    return _uniform(world, boundary)


def position_uniform_token(world: World, boundary: int, token_id: int) -> float:
    return _uniform(world, boundary, token_id)


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
    draw_kernel: str = "argmax"
    min_p: float = 0.0
    selective_noise_k: int | None = None
    gaussian_noise_std: float = 1.0
    perturb_noise_std: float = 1.0
    student_t_df: float = 3.0
    gumbel_noise_address: str = "token-id"
    gumbel_noise_scale: float = 1.0
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        if self.selective_noise_k is not None:
            if type(self.selective_noise_k) is not int or self.selective_noise_k < 1:
                raise ValueError("selective_noise_k must be positive or None")
            if self.draw_kernel == "argmax":
                raise ValueError("selective noise requires a noisy kernel")
        if self.draw_kernel not in DRAW_KERNELS:
            raise ValueError("unsupported draw kernel")
        for name in ("min_p", "gaussian_noise_std", "perturb_noise_std", "gumbel_noise_scale", "student_t_df", "repeat_penalty", "presence_penalty", "frequency_penalty"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.min_p <= 1 or min(self.gaussian_noise_std, self.perturb_noise_std, self.gumbel_noise_scale) < 0 or min(self.student_t_df, self.repeat_penalty) <= 0:
            raise ValueError("invalid sampler scale")
        if type(self.repeat_last_n) is not int or self.repeat_last_n < -1:
            raise ValueError("repeat_last_n must be -1 or nonnegative")
        if self.gumbel_noise_address not in ("token-id", "model-rank"):
            raise ValueError("unsupported Gumbel address")
        excluded = _tokens(self.excluded_token_ids)
        biases = tuple((i, float(v)) for i, v in self.biases)
        if any(type(i) is not int or i < 0 or not math.isfinite(v) for i, v in biases):
            raise ValueError("biases must contain nonnegative token IDs and finite values")
        if len(set(excluded)) != len(excluded) or len({i for i, _ in biases}) != len(biases):
            raise ValueError("duplicate token ID in policy")
        object.__setattr__(self, "excluded_token_ids", excluded)
        object.__setattr__(self, "biases", biases)

    def distribution(self, logits: Sequence[float], prefix_token_ids: Sequence[int]) -> Distribution:
        """Score and filter candidates in float64 with production ordering.

        Apply history penalties, fixed biases and production-order filters.
        Softmax is optional evidence; eligibility requires only scores.
        """
        values = tuple(float(v) for v in logits)
        if not values or any(not math.isfinite(v) for v in values):
            raise ValueError("logits must be a finite nonempty vector")
        _tokens(prefix_token_ids)
        if any(i >= len(values) for i in self.excluded_token_ids) or any(i >= len(values) for i, _ in self.biases):
            raise ValueError("policy token ID exceeds vocabulary")
        history = tuple(prefix_token_ids) if self.repeat_last_n == -1 else tuple(prefix_token_ids)[-self.repeat_last_n:] if self.repeat_last_n else ()
        if (self.repeat_penalty != 1 or self.presence_penalty != 0 or self.frequency_penalty != 0) and any(i >= len(values) for i in history):
            raise ValueError("history token ID exceeds vocabulary")
        counts = Counter(history)
        values = tuple((v * self.repeat_penalty if v < 0 else v / self.repeat_penalty) - self.presence_penalty - counts[i] * self.frequency_penalty if i in counts else v for i, v in enumerate(values))
        bias = dict(self.biases)
        eligible = [i for i in range(len(values)) if i not in self.excluded_token_ids]
        if not eligible:
            raise ValueError("policy excluded every candidate")
        if self.temperature == 0:
            winner = min(eligible, key=lambda i: (-(values[i] + bias.get(i, 0.0)), i))
            return Distribution((winner,), (values[winner] + bias.get(winner, 0.0),))
        scores = {i: (values[i] + bias.get(i, 0.0)) / float(self.temperature) for i in eligible}
        if any(not math.isfinite(v) for v in scores.values()):
            raise ValueError("policy produced non-finite scores")
        ids = sorted(eligible, key=lambda i: (-scores[i], i))[:self.top_k] if self.top_k is not None else eligible.copy()
        if self.min_p > 0:
            cutoff = max(scores[i] for i in ids) + math.log(self.min_p)
            ids = [i for i in ids if scores[i] >= cutoff]
        return Distribution(tuple(ids), tuple(scores[i] for i in ids))


@dataclass(frozen=True)
class Distribution:
    ids: tuple[int, ...]
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.scores):
            raise ValueError("candidate arrays must have equal nonzero length")
        if len(set(self.ids)) != len(self.ids) or any(type(i) is not int or i < 0 for i in self.ids):
            raise ValueError("candidate IDs must be unique nonnegative integers")
        if any(not math.isfinite(s) for s in self.scores):
            raise ValueError("scores must be finite")

    @property
    def probabilities(self) -> tuple[float, ...]:
        maximum = max(self.scores)
        weights = tuple(math.exp(score - maximum) for score in self.scores)
        total = sum(weights)
        return tuple(weight / total for weight in weights)


def draw(distribution: Distribution, world: World, boundary: int, kernel: str, *, policy: Policy | None = None, model_ranks: dict[int, int] | None = None) -> int:
    policy = policy or Policy(draw_kernel=kernel)
    if kernel not in DRAW_KERNELS:
        raise ValueError("unsupported draw kernel")
    noisy_ids = set(distribution.ids)
    if policy.selective_noise_k is not None:
        indices = sorted(range(len(distribution.ids)),
                         key=lambda j: (-distribution.scores[j], distribution.ids[j]))
        noisy_ids = {distribution.ids[j] for j in indices[:policy.selective_noise_k]}
    def rank(index):
        token_id = distribution.ids[index]
        scale = policy.gumbel_noise_scale if kernel == "gumbel-max" else policy.gaussian_noise_std if kernel == "gaussian-max" else policy.perturb_noise_std
        if kernel == "argmax" or token_id not in noisy_ids or scale == 0:
            return distribution.scores[index], -token_id
        def uniform(lane=0):
            if kernel == "gumbel-max":
                if policy.gumbel_noise_address == "token-id":
                    return position_uniform_token(world, boundary, token_id)
                if model_ranks is None:
                    raise ValueError("model ranks required")
                payload = f"{RNG_SCHEME}:gumbel-max:model-rank:{world.seed}:{world.stream_fingerprint}:{boundary}:{model_ranks[token_id]}"
            else:
                family = "gaussian-max" if kernel == "gaussian-max" else "perturb-max"
                payload = f"{RNG_SCHEME}:{family}:{world.seed}:{world.stream_fingerprint}:{boundary}:{token_id}:{lane}"
            value = int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big")
            return (value + .5) / float(1 << 64) if kernel in ("gumbel-max", "gaussian-max") else ((value >> 12) + .5) / float(1 << 52)
        u = uniform()
        if kernel == "gumbel-max":
            noise = math.inf if u == 1 else -math.log(-math.log(u))
        elif kernel == "gaussian-max":
            noise = math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * uniform(1))
        elif kernel == "logistic-max":
            noise = math.sqrt(3) / math.pi * (math.log(u) - math.log1p(-u))
        elif kernel == "laplace-max":
            noise = (math.log(2*u) if u < .5 else -math.log(2*(1-u))) / math.sqrt(2)
        elif kernel == "uniform-max":
            noise = math.sqrt(3) * (2*u-1)
        else:
            noise = _student_noise(uniform, policy.student_t_df)
        result = distribution.scores[index] + scale * noise
        if not math.isfinite(result) and not (kernel == "gumbel-max" and scale == 1):
            raise ValueError("noise produced non-finite ranking scores")
        return result, -token_id
    return distribution.ids[max(range(len(distribution.ids)), key=rank)]


def _student_noise(uniform, df):
    if df == 3:
        normals = []
        for lane in (0, 2):
            radius = math.sqrt(-2 * math.log(uniform(lane)))
            angle = 2 * math.pi * uniform(lane+1)
            normals.extend((radius * math.cos(angle), radius * math.sin(angle)))
        return normals[0] / math.sqrt(sum(n*n for n in normals[1:]))
    lane = 0
    def next_uniform():
        nonlocal lane
        value = uniform(lane)
        lane += 1
        return value
    def normal():
        return math.sqrt(-2 * math.log(next_uniform())) * math.cos(2 * math.pi * next_uniform())
    numerator = normal()
    shape = df / 2
    if shape == 0:
        log_gamma = -math.inf
    else:
        boosted = max(shape, shape+1 if shape < 1 else shape)
        d = boosted - 1/3
        c = 1 / math.sqrt(9*d)
        for _ in range(128):
            z = normal()
            base = 1+c*z
            if base <= 0:
                continue
            cube = base * base * base
            u = next_uniform()
            if u < 1-.0331*z**4 or math.log(u) < .5*z*z+d*(1-cube+math.log(cube)):
                log_gamma = math.log(d)+math.log(cube)
                break
        else:
            raise ValueError("Student-t gamma draw did not converge")
        if shape < 1:
            log_gamma += math.log(next_uniform()) / shape
    if numerator == 0:
        return 0.0
    magnitude = math.log(abs(numerator)) - .5*(math.log(2)+log_gamma-math.log(df)) - .5*math.log(3)
    if magnitude > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return math.copysign(math.inf, numerator)
    return math.copysign(math.exp(magnitude), numerator)


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
    sampling_boundary: int
    prefix_token_ids: tuple[int, ...]
    distribution: Distribution
    proposal_token_id: int


def observe(backend: Backend, branch: Branch) -> Observation:
    if branch.state.ended:
        raise ValueError("no live boundary after termination")
    prefix = branch.state.token_ids
    logits = tuple(backend.logits(prefix))
    distribution = branch.policy.distribution(logits, prefix)
    ranks = {i: rank for rank, i in enumerate(sorted(range(len(logits)), key=lambda i: (-logits[i], i)), 1)}
    boundary = branch.state.boundary
    return Observation(branch.state.boundary, boundary, prefix, distribution,
                       draw(distribution, branch.world, boundary, branch.policy.draw_kernel, policy=branch.policy, model_ranks=ranks))


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
    """Replace the exact seed, leaving prefix, policy, and boundary intact."""
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
