"""Model-free projection of the compact episode database."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .core.actions import Accept, EndGeneration, Hold, Phrase, SelectRawRank, Write
from .core.errors import EditorError
from .episode_lineage import EpisodeRelation, LineageNode, LineageView
from .episode_lineage_source import EpisodeLineageReader, build_lineage_view
from .episode_replay_source import build_source_replay_recipe, final_sampling, replay_procedure
from .episode_store import EpisodeStore


@dataclass(frozen=True)
class EpisodeProjection:
    episode_id: str
    text: str
    annotations: tuple[str, ...]
    status: str
    terminal_reason: str | None


_TEACHER_ACTION_KINDS = {"accept", "select-raw-rank", "write", "check-phrase", "force-phrase"}


def _recompute_missing_metrics(
    store: EpisodeStore,
    episode_id: str,
    episode: dict[str, Any],
    tokens: list[dict[str, Any]],
    required_by_boundary: dict[int, frozenset[str]],
    *,
    backend: Any | None,
    guidance_backend: Any | None,
) -> None:
    """Replay recorded controls/actions to fill requested report values in memory."""
    missing = {
        (int(token["boundary"]), int(token["token_id"])): token
        for token in tokens
        if token["realized_visible"] and any(
            token.get(field) is None
            for field in required_by_boundary.get(int(token["boundary"]), ())
        )
    }
    if not missing:
        return

    from .core.sampler_config import SamplerConfig
    from .episode_backend_loader import load_cfg_guidance_backend, load_episode_backend
    from .episode_engine import EpisodeEngine
    from .run_loop import ReplayContext, ReplayPlan, run_plan
    from .spr_recipe import ReplayControlPolicy, ReplayPlacement, compose_replay_plan

    owned_backend = backend is None
    owned_guidance = False
    if backend is None:
        if not episode["backend"].get("model_path"):
            raise EditorError("recorded model location is unavailable for metric replay")
        # The parser supplies backend load defaults; saved load options take priority.
        from .episode_cli import build_parser

        class NoPrompt:
            def read(self, _message: str) -> None:
                return None

            def write(self, _message: str) -> None:
                return None

        args = build_parser().parse_args([])
        backend, provenance, _ = load_episode_backend(
            args, episode, NoPrompt(), use_saved=True
        )

    try:
        saved = episode["backend"]
        actual = backend.provenance(include_model_sha256=True)
        keys = ("backend", "model_sha256", "vocabulary_size", "model_type")
        if saved.get("backend") in {"llama.cpp", "transformers"} and not saved.get("model_sha256"):
            raise EditorError("saved episode has no model checksum for metric replay")
        matched = [key for key in keys if saved.get(key) is not None]
        if not matched or any(saved[key] != actual.get(key) for key in matched):
            raise EditorError("recorded model identity differs from the loaded model")
        if guidance_backend is None and any(
            SamplerConfig.from_record(row["sampling"]).cfg_unconditional_prompt is not None
            for row in store.sampler_segments(episode_id)
        ):
            if not owned_backend:
                raise EditorError("guidance backend is required to replay CFG report metrics")
            guidance_backend = load_cfg_guidance_backend(args, provenance)
            owned_guidance = True
        if guidance_backend is not None:
            guidance = guidance_backend.provenance(include_model_sha256=True)
            if any(saved[key] != guidance.get(key) for key in matched):
                raise EditorError("recorded model identity differs from the guidance model")

        initial = store.sampling_segment(episode_id, 0)
        budgets = store.budget_segments(episode_id)
        first_budget = budgets[0]
        engine = EpisodeEngine(
            backend,
            sampling=SamplerConfig.from_record(initial["sampling"]),
            initial_text=episode["initial_text"],
            initial_token_ids=episode["initial_token_ids"],
            max_tokens=first_budget["max_tokens"],
            stream_fingerprint=initial["stream_fingerprint"],
            coordinate_offset=int(initial["coordinate_offset"]),
            guidance_backend=guidance_backend,
        )
        engine.checkpoint_boundary = first_budget["checkpoint_boundary"]

        def capture(observation: Any, token_id: int) -> None:
            token = missing.get((observation.boundary, token_id))
            if token is None:
                return
            stats = observation.statistics
            fields = required_by_boundary[observation.boundary]
            if "raw_model_nll" in fields and token.get("raw_model_nll") is None:
                token["raw_model_nll"] = stats.raw_nll(token_id)
            if "raw_rank" in fields and token.get("raw_rank") is None:
                token["raw_rank"] = stats.raw_rank(token_id)
            if "policy_rank" in fields and token.get("policy_rank") is None:
                token["policy_rank"] = stats.policy_rank(token_id)

        engine._metric_sink = capture

        class ProjectionTarget:
            identifier = episode_id

            def __init__(self) -> None:
                self.engine = engine

            def begin(self) -> int:
                return 0

            def set_sampler(self, sampling: SamplerConfig) -> None:
                engine.sampling = sampling

            def apply(self, action: Any, *, expectation: Any, divergence_policy: str, replay: bool) -> Any:
                boundary = engine.boundary
                budget = next(row for row in reversed(budgets) if int(row["start_boundary"]) <= boundary)
                engine.max_tokens = budget["max_tokens"]
                engine.checkpoint_boundary = budget["checkpoint_boundary"]
                return engine.apply(action, expectation=expectation, divergence_policy=divergence_policy, replay=replay)

            def record_replay(self, *_args: Any) -> None:
                return None

            def record_instruction_rejected(self, *_args: Any) -> None:
                return None

            def complete(self, _had_tape: bool) -> None:
                return None

        recipe = build_source_replay_recipe(store, episode_id)
        plan = compose_replay_plan(
            recipe, ReplayPlacement.SOURCE_ROOT, ReplayControlPolicy.FOLLOW_SOURCE
        )
        if plan.incomplete_handoff_reason is not None:
            raise EditorError(plan.incomplete_handoff_reason)
        target = ProjectionTarget()
        replayed = 0
        while replayed < len(plan.steps):
            if engine.checkpointed:
                budget = next(
                    row for row in reversed(budgets)
                    if int(row["start_boundary"]) <= engine.boundary
                )
                if (
                    budget["checkpoint_boundary"] is not None
                    and int(budget["checkpoint_boundary"]) <= engine.boundary
                ):
                    raise EditorError("recorded budget cannot resume metric replay")
                engine.resume(max_tokens=budget["max_tokens"])
                engine.checkpoint_boundary = budget["checkpoint_boundary"]
            chunk = ReplayPlan(
                steps=plan.steps[replayed:],
                follow_source_sampling=plan.follow_source_sampling,
                final_sampling=plan.final_sampling,
                context=ReplayContext(
                    sampling=plan.context.sampling[replayed:],
                    origins=plan.context.origins[replayed:],
                ),
            )
            result = run_plan(target, divergence_policy="handoff", tape=chunk)
            if result.handed_off or result.replayed_actions == 0:
                raise EditorError("recorded episode could not be replayed exactly for report metrics")
            replayed += result.replayed_actions
        if any(
            token.get(field) is None
            for token in missing.values()
            for field in required_by_boundary[int(token["boundary"])]
        ):
            raise EditorError("recorded episode could not be replayed exactly for report metrics")
    finally:
        if owned_guidance and guidance_backend is not None:
            close_guidance = getattr(guidance_backend, "close", None)
            if callable(close_guidance):
                close_guidance()
        if owned_backend and backend is not None:
            close_backend = getattr(backend, "close", None)
            if callable(close_backend):
                close_backend()


def _probability(value: float) -> str:
    """Keep ordinary probabilities readable without erasing tiny values."""
    return f"{value:.6g}"


def _token_evidence_footnote(
    token: dict[str, Any],
    action: dict[str, Any],
    *,
    full_evidence: bool,
    with_model_probs: bool,
) -> str:
    parts = [
        f"token={str(token['text'])!r}",
        f"id={int(token['token_id'])}",
        f"teacher={action['kind']}",
    ]
    if full_evidence:
        parts.extend(
            (
                "proposal="
                + ("agree" if bool(token["proposal_agreement"]) else "different"),
                f"nll={float(token['raw_model_nll']):.4f}",
                f"raw-rank={int(token['raw_rank'])}",
                f"policy-rank={int(token['policy_rank'])}",
            )
        )
    if with_model_probs:
        raw_probability = math.exp(-float(token["raw_model_nll"]))
        decoder_probability = float(token["decoder_probability"])
        parts.append(
            "MODEL["
            f"raw-p={_probability(raw_probability)}, "
            f"decoder-p={_probability(decoder_probability)}"
            "]"
        )
    return " · ".join(parts)


def project_fork_map(store: EpisodeStore, episode_id: str) -> str:
    """Render every legal visible-token fork boundary as an inline cut point.

    ``|N|`` is intentionally the same absolute boundary accepted by ``f N``:
    forking there preserves all generated tokens to its left and resumes before
    the token to its right.  Boundary zero therefore restarts from the original
    episode prompt/context.
    """
    episode = store.get_episode(episode_id)
    visible = [
        token for token in store.tokens(episode_id) if bool(token["realized_visible"])
    ]
    pieces = [str(episode["initial_text"])]
    for token in visible:
        pieces.append(f"|{int(token['boundary'])}|")
        pieces.append(str(token["text"]))
    live_boundary = int(visible[-1]["boundary"]) + 1 if visible else 0
    pieces.append(f"|{live_boundary}|")
    return "".join(pieces)


def project_live_fork_map(
    prompt: str,
    visible_token_ids: Sequence[int],
    backend: Any,
) -> str:
    """Render the root-relative fork map for an in-memory live branch."""
    pieces = [str(prompt)]
    for boundary, token_id in enumerate(visible_token_ids):
        pieces.append(f"|{boundary}|")
        pieces.append(str(backend.token_text(int(token_id))))
    pieces.append(f"|{len(visible_token_ids)}|")
    return "".join(pieces)


def _lineage_label(
    record: EpisodeRelation,
    *,
    selected_episode_id: str,
    relation: str,
) -> str:
    marker = "* " if record.episode_id == selected_episode_id else ""
    count = record.visible_token_count
    token_label = "token" if count == 1 else "tokens"
    label = (
        f"{marker}{record.episode_id} [{record.status}]"
        f" · {relation} · {count} {token_label}"
    )
    terminal_reason = record.terminal_reason
    if terminal_reason:
        label += f" · terminal={terminal_reason}"
    return label


def _append_lineage_tree(
    node: LineageNode,
    lines: list[str],
    *,
    selected_episode_id: str,
    prefix: str = "",
    connector: str = "",
) -> None:
    boundary = node.record.fork_boundary
    relation = (
        "root"
        if not connector
        else f"fork@{int(boundary)}" if boundary is not None else "fork@?"
    )
    lines.append(
        prefix
        + connector
        + _lineage_label(
            node.record,
            selected_episode_id=selected_episode_id,
            relation=relation,
        )
    )
    children = node.children
    for index, child in enumerate(children):
        is_last = index == len(children) - 1
        _append_lineage_tree(
            child,
            lines,
            selected_episode_id=selected_episode_id,
            prefix=prefix + ("   " if is_last else "│  ")
            if connector
            else prefix,
            connector="└─ " if is_last else "├─ ",
        )


def render_lineage(view: LineageView) -> str:
    """Render a typed lineage view without knowing how it was loaded."""

    selected_episode_id = view.selected_record.episode_id
    lines = [
        "--- lineage ---",
        f"selected: {selected_episode_id}",
        f"family root: {view.ordinary_family_root_id or '-'}",
        "fork family:",
    ]
    tree = view.ordinary_fork_tree
    if tree is None:
        lines.append("  (no ordinary fork family)")
    else:
        _append_lineage_tree(
            tree,
            lines,
            selected_episode_id=selected_episode_id,
        )

    replays = view.related_replays
    if replays:
        lines.append("replays:")
        for replay in replays:
            source = replay.spr_source_id or "-"
            context = replay.parent_id or "-"
            lines.append(
                "  "
                + _lineage_label(
                    replay,
                    selected_episode_id=selected_episode_id,
                    relation=f"source={source} · context={context}",
                )
            )

    replay_derived_forks = view.replay_derived_forks
    if replay_derived_forks:
        lines.append("forks from replay contexts:")
        for child in replay_derived_forks:
            parent = child.parent_id or "-"
            boundary = child.fork_boundary
            relation = f"parent={parent}"
            if boundary is not None:
                relation += f" · fork@{int(boundary)}"
            lines.append(
                "  "
                + _lineage_label(
                    child,
                    selected_episode_id=selected_episode_id,
                    relation=relation,
                )
            )
    return "\n".join(lines)


def project_lineage(reader: EpisodeLineageReader, episode_id: str) -> str:
    """Load and render the selected episode's fork family and replays."""

    return render_lineage(build_lineage_view(reader, episode_id))


def project_episode(
    store: EpisodeStore,
    episode_id: str,
    *,
    include_initial: bool = True,
    annotations: str = "none",
    with_loss: bool = False,
    with_rank: bool = False,
    with_policy_rank: bool = False,
    full_evidence: bool = False,
    with_model_probs: bool = False,
    with_lineage: bool = False,
    backend: Any | None = None,
    guidance_backend: Any | None = None,
) -> EpisodeProjection:
    if annotations not in {"none", "inline", "footnotes"}:
        raise EditorError("annotations must be none, inline, or footnotes")
    episode = store.get_episode(episode_id)
    base = str(episode["initial_text"]) if include_initial else ""
    if not (
        annotations != "none"
        or with_loss
        or with_rank
        or with_policy_rank
        or full_evidence
        or with_model_probs
    ):
        # Full-sequence detokenization is authoritative for the seamless view.
        text = base + str(episode["visible_text"])
        if with_lineage:
            text += "\n\n" + project_lineage(store, episode_id)
        return EpisodeProjection(
            episode_id=episode_id,
            text=text,
            annotations=(),
            status=str(episode["status"]),
            terminal_reason=episode.get("terminal_reason"),
        )
    tokens = store.tokens(episode_id)
    actions = store.actions(episode_id)
    interactions = store.interactions(episode_id)
    actions_by_ordinal = {int(action["ordinal"]): action for action in actions}
    required_by_boundary: dict[int, frozenset[str]] = {}
    for token in tokens:
        if not token["realized_visible"]:
            continue
        fields: set[str] = set()
        teacher = (
            (action := actions_by_ordinal.get(int(token["action_ordinal"]))) is not None
            and action["kind"] in _TEACHER_ACTION_KINDS
        )
        if (with_loss and not full_evidence) or (teacher and (full_evidence or with_model_probs)):
            fields.add("raw_model_nll")
        if (with_rank and not full_evidence) or (teacher and full_evidence):
            fields.add("raw_rank")
        if (with_policy_rank and not full_evidence) or (teacher and full_evidence):
            fields.add("policy_rank")
        if fields:
            required_by_boundary[int(token["boundary"])] = frozenset(fields)
    if required_by_boundary:
        _recompute_missing_metrics(
            store, episode_id, episode, tokens, required_by_boundary,
            backend=backend, guidance_backend=guidance_backend,
        )
    visible = [token for token in tokens if bool(token["realized_visible"])]
    notes_by_boundary: dict[int, list[str]] = {}
    for interaction in interactions:
        if interaction["kind"] not in {"note-before", "note-after"}:
            continue
        text = interaction["payload"].get("text")
        if isinstance(text, str) and text:
            notes_by_boundary.setdefault(int(interaction["boundary"]), []).append(text)

    annotation_notes: list[str] = []
    footnotes: list[str] = []
    pieces: list[str] = [base]
    for token in visible:
        boundary = int(token["boundary"])
        note_values = notes_by_boundary.get(boundary, [])
        if annotations == "inline" and note_values:
            pieces.append("[" + " | ".join(note_values) + "]")
        elif annotations == "footnotes" and note_values:
            for note in note_values:
                annotation_notes.append(note)
                footnotes.append(note)
                pieces.append(f"[^{len(footnotes)}]")
        pieces.append(str(token["text"]))

        action = actions_by_ordinal.get(int(token["action_ordinal"]))
        if (
            action is not None
            and action["kind"] in _TEACHER_ACTION_KINDS
            and (full_evidence or with_model_probs)
        ):
            footnotes.append(
                _token_evidence_footnote(
                    token,
                    action,
                    full_evidence=full_evidence,
                    with_model_probs=with_model_probs,
                )
            )
            pieces.append(f"[^{len(footnotes)}]")

        evidence: list[str] = []
        if with_loss and not full_evidence:
            evidence.append(f"nll={float(token['raw_model_nll']):.4f}")
        if with_rank and not full_evidence:
            evidence.append(f"raw-rank={int(token['raw_rank'])}")
        if with_policy_rank and not full_evidence:
            evidence.append(f"policy-rank={int(token['policy_rank'])}")
        if evidence:
            pieces.append("{" + ", ".join(evidence) + "}")

    if footnotes:
        pieces.append("\n\n")
        pieces.extend(
            f"[^{index}]: {note}\n" for index, note in enumerate(footnotes, 1)
        )
    if with_lineage:
        pieces.append("\n\n")
        pieces.append(project_lineage(store, episode_id))
    return EpisodeProjection(
        episode_id=episode_id,
        text="".join(pieces),
        annotations=tuple(annotation_notes),
        status=str(episode["status"]),
        terminal_reason=episode.get("terminal_reason"),
    )


def _procedure_text(text: str) -> str:
    """Escape control characters without losing Unicode or leading spaces."""
    import json
    return json.dumps(text, ensure_ascii=False)[1:-1].replace(r'\"', '"')


def project_procedure(store: EpisodeStore, episode_id: str) -> str:
    """Render the same surviving procedure used by replay, without a model."""
    from pathlib import PurePosixPath
    from .core.sampler_config import SamplerConfig

    episode = store.get_episode(episode_id)
    steps = replay_procedure(store, episode_id)
    initial = SamplerConfig.from_record(store.sampling_segment(episode_id, 0)["sampling"])
    backend = episode["backend"]
    model = backend.get("filename") or backend.get("model_path") or backend.get("model") or "unknown"
    model = PurePosixPath(str(model).replace(chr(92), "/")).name
    fields = {
        key: getattr(initial, key)
        for key in initial.__dataclass_fields__
        if key not in {"bias_rules", "bias_groups"}
    }
    lines = [
        f"MODEL   : {_procedure_text(model)}",
        f"BACKEND : {_procedure_text(str(backend.get('backend', 'unknown')))}",
        "SAMPLER : " + " ".join(f"{key}={value}" for key, value in fields.items()),
        f"P       : {_procedure_text(str(episode['initial_text']))}",
        "",
    ]
    # No source budget is imported by replay; do not pretend the last saved
    # allowance was necessarily the allowance at the start of this procedure.
    rows: list[tuple[int, str, str | None]] = []
    current = initial
    boundary = 0

    def transition(config: SamplerConfig, at: int, *, trailing: bool = False) -> None:
        nonlocal current
        changes = [
            f"{key}={value}" for key in fields
            if (value := getattr(config, key)) != getattr(current, key)
        ]
        if changes:
            rows.append((at, "q", None))
            rows.append((at, "s " + " ".join(changes), None))
            if not trailing:
                rows.append((at, "c", None))
        if config.bias_rules != current.bias_rules:
            rows.append((at, f"# Set logical bias rules to {[r.to_dict() for r in config.bias_rules]!r}", None))
        if config.bias_groups != current.bias_groups:
            rows.append((at, f"# Set bias groups to {[group.to_dict() for group in config.bias_groups]!r}", None))
        current = config

    if initial.bias_rules:
        lines.insert(3, f"RULES   : logical bias rules {[r.to_dict() for r in initial.bias_rules]!r}")
    if initial.bias_groups:
        lines.insert(4, f"GROUPS  : {[group.to_dict() for group in initial.bias_groups]!r}")

    for step in steps:
        boundary = step["boundary"]
        transition(step["sampling"], boundary)
        action = step["action"]
        tokens = step["tokens"]
        result = "".join(str(token["text"]) for token in tokens)
        comment: str | None = _procedure_text(result)
        if isinstance(action, SelectRawRank):
            command = str(action.rank)
        elif isinstance(action, Accept):
            # Legacy acceptance can only be printed as its recorded source rank.
            command = (
                str(tokens[0]["raw_rank"])
                if tokens and tokens[0].get("raw_rank") is not None
                else "accept"
            )
        elif isinstance(action, Write):
            command = ("t " if action.mode == "continuation" else "x ") + action.text
            if any(ord(char) < 32 or ord(char) == 127 for char in action.text):
                command = ("t " if action.mode == "continuation" else "x ") + _procedure_text(action.text)
                comment = "display-escaped write; control characters must be pasted literally"
            else:
                comment = None
        elif isinstance(action, Phrase):
            command = (
                "force" if action.force else "check"
            ) + ("x " if action.mode == "exact" else " ") + action.text
            if any(ord(char) < 32 or ord(char) == 127 for char in action.text):
                prefix = "forcex " if action.force and action.mode == "exact" else (
                    "force " if action.force else "checkx " if action.mode == "exact" else "check "
                )
                command = prefix + _procedure_text(action.text)
                comment = "display-escaped phrase; control characters must be pasted literally"
            else:
                comment = None
        elif isinstance(action, Hold):
            marker = {"sentence": ". ", "newline": "| ", None: ""}[action.boundary]
            command = f"h {marker}{action.limit}"
        elif isinstance(action, EndGeneration):
            command = "e!"
        else:
            raise EditorError(f"cannot render procedure action {action!r}")
        rows.append((boundary, command, comment))
        boundary += len(step["expectation"].token_ids)

    transition(final_sampling(store, episode_id), boundary, trailing=True)
    # Finite procedures return live control, never seal the destination.
    if not rows or rows[-1][1] not in {"q", "e!"} and not rows[-1][1].startswith("s "):
        rows.append((boundary, "q", None))
    width = max((len(f"{at} : {command}") for at, command, _ in rows), default=0)
    for at, command, comment in rows:
        line = f"{at} : {command}"
        lines.append(line if comment is None else line.ljust(width) + " #" + comment)
    return chr(10).join(lines)
