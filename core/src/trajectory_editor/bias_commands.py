"""Core semantic resolution for interactive manual bias edits."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from .bias_rules import BiasGroup, BiasRule, GROUP_NAME_RE
from .core.errors import EditorError


def _name(text: str) -> str:
    return text if GROUP_NAME_RE.fullmatch(text) else "term_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def resolve_target(text, bare, backend, groups):
    """Resolve a menu target directly to a core bias group."""
    semantic = text.strip() if bare else text
    if bare and semantic in groups:
        return groups[semantic]
    route = tuple(backend.tokenize(semantic, add_bos=False, special=False))
    if not route or any(
        type(token) is not int
        or not 0 <= token < backend.vocabulary_size()
        or backend.is_eog(token)
        for token in route
    ):
        raise EditorError(f"bias target {semantic!r} produced no ordinary model tokens")
    name = _name(semantic) if bare else "exact_" + hashlib.sha256(text.encode()).hexdigest()[:16]
    return groups.get(name, BiasGroup(
        name=name,
        rules=(BiasRule(routes=(route,), bias=0, mode="tail"),),
        members=(semantic,),
        surfaces=(semantic,),
    ))


def _scope(command, backend, groups):
    if command.bias_triggers is None:
        return (), None
    flags = command.bias_trigger_bare or (True,) * len(command.bias_triggers)
    routes = set()
    for text, bare in zip(command.bias_triggers, flags):
        target = resolve_target(text, bare, backend, groups)
        routes.update(route for rule in target.rules for route in rule.routes)
    until = None
    if command.bias_stop_text is not None:
        stop = backend.tokenize(command.bias_stop_text, add_bos=False, special=False)
        if len(stop) != 1:
            raise EditorError("scoped-bias stop text must tokenize to exactly one token; use until #N")
        until = stop[0]
    elif command.bias_stop_token is not None:
        until = command.bias_stop_token
    return tuple(sorted(routes)), until


def apply_bias_command(command, backend, sampling, observation, resolve_candidate):
    """Return an updated sampler and concise user-facing change labels."""
    if command.bias_status:
        return sampling, []
    groups = {group.name: group for group in sampling.bias_groups}
    triggers, until = _scope(command, backend, groups)
    rules = {rule.key: rule for rule in sampling.bias_rules}
    updates = []

    if command.bias_group_name is not None:
        name = command.bias_group_name
        existing = groups.get(name)
        members = list(existing.members) if existing else []
        surfaces = list(existing.surfaces) if existing else []
        templates = {rule.key: rule for rule in existing.rules} if existing else {}
        texts = command.bias_group_members or ()
        flags = command.bias_group_member_bare or (False,) * len(texts)
        for text, bare in zip(texts, flags):
            target = resolve_target(text, bare, backend, groups)
            templates.update((rule.key, rule) for rule in target.rules)
            members.extend(target.members)
            surfaces.extend(target.surfaces)
        if not templates:
            raise EditorError(f"group {name!r} requires at least one member")
        group = BiasGroup(
            name=name,
            rules=tuple(templates.values()),
            bias=existing.bias if existing else 0.0,
            members=tuple(dict.fromkeys(members)),
            surfaces=tuple(dict.fromkeys(surfaces)),
            enabled=existing.enabled if existing else True,
        )
        groups[name] = group
        return replace(sampling, bias_groups=tuple(groups.values())), [
            (f"Group {name!r} · {len(group.members)} members", group.bias)
        ]

    if command.bias_targets is None:
        if command.bias_last is not None:
            if command.bias_last > len(observation.prefix_token_ids):
                raise EditorError("bl requests more tokens than the current context contains")
            tokens = observation.prefix_token_ids[-command.bias_last:]
        else:
            candidate = resolve_candidate(command.search_rank)
            prefix = () if command.bias_prefix is None else tuple(
                backend.tokenize(command.bias_prefix, add_bos=False, special=False)
            )
            if command.bias_prefix is not None and not prefix:
                raise EditorError("bias prefix produced no tokens")
            tokens = (*prefix, candidate.token_id)
        template = BiasRule(routes=(tuple(tokens),), bias=0, triggers=triggers, until=until)
        old = rules[template.key].bias if template.key in rules else 0.0
        step = command.bias_amount if command.bias_amount is not None else sampling.bias_step
        amount = 0.0 if command.bias_operator == "=" else old + (
            step if command.bias_operator == "+" else -step
        )
        rules[template.key] = replace(template, bias=amount)
        updates.append(("Token bias", amount))
        return replace(sampling, bias_rules=tuple(rules.values())), updates

    flags = command.bias_target_bare or (True,) * len(command.bias_targets)
    targets = [resolve_target(text, bare, backend, groups) for text, bare in zip(command.bias_targets, flags)]
    if len({tuple(sorted(route for rule in group.rules for route in rule.routes)) for group in targets}) != len(targets):
        raise EditorError("bias targets resolve to duplicate token sequences")
    for group in targets:
        groups[group.name] = group
        if command.bias_operator == "off":
            groups[group.name] = replace(group, enabled=False)
            updates.append((f"Group {group.name!r} off", 0.0))
            continue
        step = command.bias_amount if command.bias_amount is not None else sampling.bias_step
        amount = 0.0 if command.bias_operator == "=" else group.bias + (
            step if command.bias_operator == "+" else -step
        )
        if triggers:
            for rule in group.rules:
                scoped = replace(rule, triggers=triggers, until=until, logical_target=f"group:{group.name}")
                rules[scoped.key] = replace(scoped, bias=amount)
        groups[group.name] = replace(group, bias=amount, enabled=True)
        updates.append((f"Group {group.name!r} manual", amount))
    return replace(sampling, bias_groups=tuple(groups.values()), bias_rules=tuple(rules.values())), updates
