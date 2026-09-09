"""Model-free projection of the compact episode database."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .domain import EditorError
from .episode_store import EpisodeStore


@dataclass(frozen=True)
class EpisodeProjection:
    episode_id: str
    text: str
    annotations: tuple[str, ...]
    status: str
    terminal_reason: str | None


_TEACHER_ACTION_KINDS = {"accept", "select-raw-rank", "write"}


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


def _lineage_label(
    node: dict[str, Any],
    *,
    selected_episode_id: str,
    relation: str,
) -> str:
    marker = "* " if node["episode_id"] == selected_episode_id else ""
    count = int(node["visible_tokens"])
    token_label = "token" if count == 1 else "tokens"
    label = (
        f"{marker}{node['episode_id']} [{node['status']}]"
        f" · {relation} · {count} {token_label}"
    )
    terminal_reason = node.get("terminal_reason")
    if terminal_reason:
        label += f" · terminal={terminal_reason}"
    return label


def _append_lineage_tree(
    node: dict[str, Any],
    lines: list[str],
    *,
    selected_episode_id: str,
    prefix: str = "",
    connector: str = "",
) -> None:
    boundary = node.get("fork_boundary")
    relation = (
        "root"
        if not connector
        else f"fork@{int(boundary)}" if boundary is not None else "fork@?"
    )
    lines.append(
        prefix
        + connector
        + _lineage_label(
            node,
            selected_episode_id=selected_episode_id,
            relation=relation,
        )
    )
    children = list(node.get("children") or [])
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


def project_lineage(store: EpisodeStore, episode_id: str) -> str:
    """Render the selected episode's fork family and related replays."""
    details = store.lineage(episode_id)
    lines = [
        "--- lineage ---",
        f"selected: {details['selected_episode_id']}",
        f"family root: {details['family_root_id'] or '-'}",
        "fork family:",
    ]
    tree = details.get("tree")
    if tree is None:
        lines.append("  (no ordinary fork family)")
    else:
        _append_lineage_tree(
            tree,
            lines,
            selected_episode_id=episode_id,
        )

    replays = list(details.get("replays") or [])
    if replays:
        lines.append("replays:")
        for replay in replays:
            source = replay.get("replay_source_episode_id") or "-"
            context = replay.get("parent_episode_id") or "-"
            lines.append(
                "  "
                + _lineage_label(
                    replay,
                    selected_episode_id=episode_id,
                    relation=f"source={source} · context={context}",
                )
            )

    replay_derived_forks = list(details.get("replay_derived_forks") or [])
    if replay_derived_forks:
        lines.append("forks from replay contexts:")
        for child in replay_derived_forks:
            parent = child.get("parent_episode_id") or "-"
            boundary = child.get("fork_boundary")
            relation = f"parent={parent}"
            if boundary is not None:
                relation += f" · fork@{int(boundary)}"
            lines.append(
                "  "
                + _lineage_label(
                    child,
                    selected_episode_id=episode_id,
                    relation=relation,
                )
            )
    return "\n".join(lines)


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
    from .domain import SamplingConfig
    from .episode_actions import Accept, EndGeneration, Finish, Hold, SelectRawRank, Write

    episode = store.get_episode(episode_id)
    steps = store.replay_procedure(episode_id)
    initial = SamplingConfig.from_record(store.sampling_segment(episode_id, 0)["sampling"])
    backend = episode["backend"]
    model = backend.get("filename") or backend.get("model_path") or backend.get("model") or "unknown"
    model = PurePosixPath(str(model).replace(chr(92), "/")).name
    fields = {key: getattr(initial, key) for key in initial.__dataclass_fields__}
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

    def transition(config: SamplingConfig, at: int, *, trailing: bool = False) -> None:
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
        current = config

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
            command = str(tokens[0]["raw_rank"]) if tokens else "accept"
        elif isinstance(action, Write):
            command = ("t " if action.mode == "continuation" else "x ") + action.text
            if any(ord(char) < 32 or ord(char) == 127 for char in action.text):
                command = ("t " if action.mode == "continuation" else "x ") + _procedure_text(action.text)
                comment = "display-escaped write; control characters must be pasted literally"
            else:
                comment = None
        elif isinstance(action, Hold):
            marker = {"sentence": ". ", "newline": "| ", None: ""}[action.boundary]
            command = f"h {marker}{action.limit}"
        elif isinstance(action, Finish):
            command = f"h {len(step['expectation'].token_ids)}"
            comment = (comment or "") + " [legacy finish shown as recorded span]"
        elif isinstance(action, EndGeneration):
            command = "e!"
        else:
            raise EditorError(f"cannot render procedure action {action!r}")
        rows.append((boundary, command, comment))
        boundary += len(step["expectation"].token_ids)

    transition(store.final_sampling(episode_id), boundary, trailing=True)
    # Finite procedures return live control, never seal the destination.
    if not rows or rows[-1][1] not in {"q", "e!"} and not rows[-1][1].startswith("s "):
        rows.append((boundary, "q", None))
    width = max((len(f"{at} : {command}") for at, command, _ in rows), default=0)
    for at, command, comment in rows:
        line = f"{at} : {command}"
        lines.append(line if comment is None else line.ljust(width) + " #" + comment)
    return chr(10).join(lines)
