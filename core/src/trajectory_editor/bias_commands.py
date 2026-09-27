"""Apply simple group and direct-token bias edits and render their sources."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from .bias_groups import BiasGroup, BiasMember, BiasToken
from .core.errors import EditorError


DEFAULT_BIAS_STEP = 0.5


def _adjusted(old: float, operator: str, amount: float | None) -> float:
    if operator == "off":
        return 0.0
    if operator == "=":
        if amount is None:
            raise EditorError("setting a bias requires an amount, for example =0.5")
        result = amount
    elif operator in {"+", "-"}:
        step = DEFAULT_BIAS_STEP if amount is None else amount
        if not isfinite(step) or step <= 0.0:
            raise EditorError("bias adjustment must be finite and positive")
        result = old + (step if operator == "+" else -step)
    else:
        raise EditorError("bias operator must be +, -, =, or off")
    if not isfinite(result):
        raise EditorError("bias amount must be finite")
    return result


def _token_text(backend, token_id: int) -> str:
    try:
        return repr(backend.token_text(token_id))
    except Exception:
        return "<token text unavailable>"


def _format_route(route, backend) -> str:
    return " ".join(
        f"#{token_id} {_token_text(backend, token_id)}"
        for token_id in route.token_ids
    )


def _endswith(history, prefix) -> bool:
    if len(history) < len(prefix):
        return False
    return not prefix or tuple(history[-len(prefix):]) == tuple(prefix)


def _format_contribution(item, backend) -> str:
    state = "active" if item.active else "inactive until its prefix matches"
    suffix = f" ({state})"
    if item.member_routes:
        details = []
        for member_route in item.member_routes:
            route = member_route.route
            member = repr(member_route.member_text)
            if member_route.member_literal:
                member += " [literal]"
            surfaces = ", ".join(repr(surface) for surface in route.surfaces)
            if member_route.active:
                condition = "matched"
            else:
                prefix = route.token_ids[:-1]
                rendered = " ".join(
                    f"#{token} {_token_text(backend, token)}" for token in prefix
                ) or "<empty>"
                condition = f"needs suffix [{rendered}]"
            details.append(f"{member}: {surfaces} {condition}")
        suffix += "; routes: " + "; ".join(details)
    return f"{item.amount:+g} from {item.source}{suffix}"


def apply_bias_command(command, backend, sampling, observation, resolve_candidate):
    """Return an updated sampler and concise user-facing change labels."""
    if command.bias_status or command.bias_inspect_group or command.bias_inspect_token is not None:
        return sampling, []

    groups = {group.name: group for group in sampling.bias_groups}
    token_biases = {item.token_id: item.bias for item in sampling.token_biases}

    if command.bias_group_name is not None:
        name = command.bias_group_name
        existing = groups.get(name)
        members = list(existing.members) if existing else []
        known = {(member.text, member.literal) for member in members}
        additions = []
        texts = command.bias_group_members or ()
        literal_flags = command.bias_group_member_literal or (False,) * len(texts)
        for text, literal in zip(texts, literal_flags):
            key = (text, literal)
            if key in known:
                continue
            additions.append(BiasMember.compile(text, backend, literal=literal))
            known.add(key)
        if not members and not additions:
            raise EditorError(f"group {name!r} requires at least one member")
        groups[name] = BiasGroup(
            name=name,
            members=tuple((*members, *additions)),
            bias=existing.bias if existing else 0.0,
        )
        return replace(sampling, bias_groups=tuple(groups.values())), [
            (f"Group {name!r} · {len(groups[name].members)} members", groups[name].bias)
        ]

    if command.bias_group_remove_name is not None:
        name = command.bias_group_remove_name
        if name not in groups:
            raise EditorError(f"no bias group named {name!r}")
        remove = set(zip(
            command.bias_group_remove_members or (),
            command.bias_group_remove_member_literal
            or (False,) * len(command.bias_group_remove_members or ()),
        ))
        group = groups[name]
        members = tuple(
            member for member in group.members
            if (member.text, member.literal) not in remove
        )
        missing = remove - {(member.text, member.literal) for member in group.members}
        if missing:
            names = ", ".join(repr(text) for text, _literal in sorted(missing))
            raise EditorError(f"group {name!r} does not contain: {names}")
        if not members:
            del groups[name]
            label = f"Group {name!r} removed"
        else:
            groups[name] = replace(group, members=members)
            label = f"Group {name!r} · {len(members)} members"
        return replace(sampling, bias_groups=tuple(groups.values())), [(label, 0.0)]

    if command.bias_targets is not None:
        names = tuple(name.strip() for name in command.bias_targets)
        if len(set(names)) != len(names):
            raise EditorError("bias group targets cannot contain duplicates")
        missing = [name for name in names if name not in groups]
        if missing:
            joined = ", ".join(repr(name) for name in missing)
            raise EditorError(f"unknown bias group(s): {joined}; create them with `b NAME -> {{members}}`")
        updates = []
        for name in names:
            group = groups[name]
            amount = _adjusted(group.bias, command.bias_operator, command.bias_amount)
            groups[name] = replace(group, bias=amount)
            updates.append((f"Group {name!r}", amount))
        return replace(sampling, bias_groups=tuple(groups.values())), updates

    if command.search_rank is not None or command.bias_token_id is not None:
        if command.bias_token_id is not None:
            token_id = command.bias_token_id
        else:
            candidate = resolve_candidate(command.search_rank)
            token_id = int(candidate.token_id)
        amount = _adjusted(
            token_biases.get(token_id, 0.0),
            command.bias_operator,
            command.bias_amount,
        )
        if amount:
            token_biases[token_id] = amount
        else:
            token_biases.pop(token_id, None)
        updated = tuple(BiasToken(token, bias) for token, bias in token_biases.items())
        return replace(sampling, token_biases=updated), [(f"Token #{token_id}", amount)]

    raise EditorError("bias command requires a group name or candidate rank")


def format_bias_status(sampling) -> str:
    lines = [
        f"{group.name}: {group.bias:+g}; {len(group.members)} members"
        for group in sampling.bias_groups
    ]
    if sampling.token_biases:
        lines.append("Direct token adjustments:")
        lines.extend(
            f"  #{item.token_id}: {item.bias:+g}"
            for item in sampling.token_biases
        )
    return "\n".join(lines) if lines else "No bias groups or direct token adjustments."


def format_group_report(sampling, name: str, history, backend) -> str:
    groups = {group.name: group for group in sampling.bias_groups}
    group = groups.get(name)
    if group is None:
        raise EditorError(f"no bias group named {name!r}")
    contributions = sampling.bias_contributions(history, include_inactive=True)
    history = () if history is None else tuple(history)
    lines = [
        f"Group {group.name}: {group.bias:+g} per member "
        "(one group contribution per token)"
    ]
    for member in group.members:
        lines.append(f"  {member.text!r}{' [literal]' if member.literal else ''}")
        by_token: dict[int, list] = {}
        for route in member.routes:
            by_token.setdefault(route.token_ids[-1], []).append(route)
        for token_id, routes in sorted(by_token.items()):
            active_routes = {
                route.token_ids for route in routes
                if _endswith(history, route.token_ids[:-1])
            }
            state = "active now" if active_routes else "inactive now"
            lines.append(
                f"    target #{token_id} {_token_text(backend, token_id)}: "
                f"{group.bias:+g} ({state})"
            )
            for route in routes:
                route_state = "matches" if route.token_ids in active_routes else "waits"
                surface = ", ".join(repr(text) for text in route.surfaces)
                lines.append(
                    f"      {route_state}: {surface} -> {_format_route(route, backend)}"
                )
            other = [item for item in contributions if item.token_id == token_id]
            active_total = sum(item.amount for item in other if item.active)
            if other:
                lines.append(f"      token total now: {active_total:+g}")
                for item in other:
                    lines.append(f"        {_format_contribution(item, backend)}")
    return "\n".join(lines)


def format_token_report(sampling, token_id: int, history, backend) -> str:
    if token_id < 0 or token_id >= backend.vocabulary_size():
        raise EditorError("token ID is outside the model vocabulary")
    contributions = tuple(
        item for item in sampling.bias_contributions(history, include_inactive=True)
        if item.token_id == token_id
    )
    active = [item for item in contributions if item.active]
    active_total = sum(item.amount for item in active)
    lines = [
        f"Token #{token_id} {_token_text(backend, token_id)}",
        f"  active total: {active_total:+g}",
    ]
    if active:
        lines.append("  active sources:")
        lines.extend(f"    {_format_contribution(item, backend)}" for item in active)
    else:
        lines.append("  no active bias sources")
    inactive = [item for item in contributions if not item.active]
    if inactive:
        lines.append("  phrase sources that are not active at this context:")
        lines.extend(f"    {_format_contribution(item, backend)}" for item in inactive)
    return "\n".join(lines)
