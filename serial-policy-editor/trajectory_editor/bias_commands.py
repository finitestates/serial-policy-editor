"""Shared semantic resolution and steering edits, independent of the UI."""

from dataclasses import replace
import hashlib

from .bias_catalog import CompileOptions, generate_forms
from .bias_rules import BiasGroup, BiasRule, GROUP_NAME_RE, routes_for_catalog_entry
from .domain import EditorError
from .group_control import GroupControl, control_span, estimate_baseline


def _name(text):
    return text if GROUP_NAME_RE.fullmatch(text) else "term_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def _entry_group(entry):
    routes = getattr(entry, "runtime_routes", ()) or entry.routes
    return BiasGroup(_name(entry.name), routes_for_catalog_entry(entry, 0), members=entry.members or (entry.source or entry.name,),
                     surfaces=tuple(dict.fromkeys(text for route in routes for text in route.texts)),
                     learnable=False)


def resolve_target(text, bare, backend, groups, catalog=None):
    semantic = text.strip() if bare else text
    if bare and semantic in groups:
        return groups[semantic]
    entry = None
    if bare and semantic.startswith("@"):
        if catalog is None:
            raise EditorError(f"catalog reference {semantic!r} requires --bias-catalog")
        entry = catalog.require(semantic[1:])
    elif bare and catalog is not None:
        entry = catalog.resolve(semantic)
    if entry is not None:
        group = _entry_group(entry)
        return groups.get(group.name, group)
    # Quoted text is exact; ordinary text gets the same surface expansion as
    # canonical YAML compilation. No vocabulary enumeration is performed.
    forms = generate_forms(semantic, CompileOptions()) if bare else (text,)
    routes = {}
    accepted_forms = []
    for form in forms:
        route = tuple(backend.tokenize(form, add_bos=False, special=False))
        if not route:
            continue
        if any(type(t) is not int or not 0 <= t < backend.vocabulary_size() or backend.is_eog(t) for t in route):
            raise EditorError("bias target must produce ordinary model tokens")
        # Backends can normalize initial spacing. Reject unsupported generated
        # variants rather than silently mapping them to an unknown token.
        rendered = backend.render(list(route), special=False)
        if bare and rendered != form and rendered.strip() != form.strip():
            continue
        routes[route] = BiasRule(routes=(route,), bias=0, mode="tail" if len(route) == 1 else "path")
        accepted_forms.append(form)
    if not routes:
        route = tuple(backend.tokenize(text, add_bos=False, special=False))
        if not route or any(type(t) is not int or not 0 <= t < backend.vocabulary_size() or backend.is_eog(t) for t in route):
            raise EditorError(f"bias target {semantic!r} produced no matching token routes")
        routes[route] = BiasRule(routes=(route,), bias=0, mode="path" if len(route) > 1 else "tail")
        accepted_forms = [text]
    name = _name(semantic) if bare else "exact_" + hashlib.sha256(text.encode()).hexdigest()[:16]
    return groups.get(name, BiasGroup(name, tuple(routes.values()), members=(semantic,),
                                     surfaces=tuple(accepted_forms), learnable=False))


def _scope(command, backend, groups, catalog):
    if command.bias_triggers is None:
        return (), None
    flags = command.bias_trigger_bare or (True,) * len(command.bias_triggers)
    routes = set()
    for text, bare in zip(command.bias_triggers, flags):
        target = resolve_target(text, bare, backend, groups, catalog)
        routes.update(route for rule in target.rules for route in rule.routes)
    until = command.bias_until
    if command.bias_stop_text is not None:
        stop = backend.tokenize(command.bias_stop_text, add_bos=False, special=False)
        if len(stop) != 1:
            raise EditorError("scoped-bias stop text must tokenize to exactly one token; use until #N")
        until = stop[0]
    elif command.bias_stop_token is not None:
        until = command.bias_stop_token
    return tuple(sorted(routes)), until


def apply_bias_command(command, backend, sampling, observation, resolve_candidate, catalog=None, *, level=1.):
    """Return a complete sampler edit and human-readable transaction records."""
    if command.bias_status:
        return sampling, []
    groups = {g.name: g for g in sampling.bias_groups}
    if command.bias_group_name is not None:
        name = command.bias_group_name
        existing = groups.get(name)
        if existing is None and catalog is not None:
            entry = catalog.resolve(name)
            if entry is not None and entry.kind == "group":
                existing = _entry_group(entry)
        rules = {r.key: r for r in existing.rules} if existing else {}
        members = list(existing.members) if existing else []
        surfaces = list(existing.surfaces) if existing else []
        texts = command.bias_group_members or ()
        flags = command.bias_group_member_bare or (False,) * len(texts)
        for text, bare in zip(texts, flags):
            target = resolve_target(text, bare, backend, groups, catalog)
            rules.update((r.key, r) for r in target.rules)
            members.extend(target.members)
            surfaces.extend(target.surfaces)
        group = BiasGroup(name, tuple(rules.values()), bias=existing.bias if existing else 0.,
                          members=tuple(dict.fromkeys(members)), surfaces=tuple(dict.fromkeys(surfaces)),
                          enabled=existing.enabled if existing else True, learnable=False)
        groups[name] = group
        return replace(sampling, bias_groups=tuple(groups.values())), [
            ("bias-group", {"group": group.to_dict()}, f"Group {name!r} · {len(group.members)} members", group.bias)]

    triggers, until = _scope(command, backend, groups, catalog)
    controls = {c.key: c for c in sampling.group_controls}
    rules = {r.key: r for r in sampling.bias_rules}
    updates = []
    if command.bias_targets is None:
        # Exact rank/tail adjustments retain the familiar manual interface.
        if command.bias_last is not None:
            if command.bias_last > len(observation.prefix_token_ids):
                raise EditorError("bl requests more tokens than the current context contains")
            tokens = observation.prefix_token_ids[-command.bias_last:]
        else:
            candidate = resolve_candidate(command.search_rank)
            prefix = () if command.bias_prefix is None else tuple(backend.tokenize(command.bias_prefix, add_bos=False, special=False))
            if command.bias_prefix is not None and not prefix:
                raise EditorError("bias prefix produced no tokens")
            tokens = (*prefix, candidate.token_id)
        template = BiasRule(routes=(tuple(tokens),), bias=0, triggers=triggers, until=until)
        old = rules[template.key].bias if template.key in rules else 0.
        step = command.bias_amount if command.bias_amount is not None else sampling.bias_step
        amount = 0. if command.bias_operator == "=" else old + (step if command.bias_operator == "+" else -step)
        rules[template.key] = replace(template, bias=amount)
        updates.append(("bias-rule", {"previous": old, "rule": rules[template.key].to_dict()}, "Token bias", amount))
    else:
        flags = command.bias_target_bare or (True,) * len(command.bias_targets)
        targets = [resolve_target(t, bare, backend, groups, catalog) for t, bare in zip(command.bias_targets, flags)]
        if len({tuple(sorted(route for r in g.rules for route in r.routes)) for g in targets}) != len(targets):
            raise EditorError("bias targets resolve to duplicate token sequences")
        for group in targets:
            groups[group.name] = group
            key = group.name, triggers, until
            identity = f"group:{group.name}"
            matching = [k for k, r in rules.items() if r.logical_target == identity and r.triggers == triggers and r.until == until]
            if command.bias_operator == "off":
                controls.pop(key, None)
                for k in matching:
                    rules.pop(k)
                if not triggers:
                    groups[group.name] = replace(group, bias=0., learnable=False)
                updates.append(("group-control", {"group": group.name, "enabled": False}, f"Group {group.name!r} off", 0.))
            elif command.bias_amount is None:
                direction = {"+": "promote", "-": "suppress", "=": "maintain"}[command.bias_operator]
                previous = controls.get(key)
                history_start = len(observation.prefix_token_ids) - observation.boundary
                gate = GroupControl(group.name, direction, 0., triggers=triggers, until=until, history_start=history_start)
                span = control_span(gate, observation.prefix_token_ids, observation.statistics.boundaries)
                if span is None:
                    span = ()
                baseline = estimate_baseline(group, span, observation.statistics.policy_probabilities,
                                             getattr(observation.statistics, "render_tokens", None))
                if previous is not None and command.bias_operator != "=":
                    baseline = previous.baseline_rate
                control = GroupControl(group.name, direction, baseline, level=level, triggers=triggers, until=until, history_start=history_start)
                controls[key] = control
                for k in matching:
                    rules.pop(k)
                if not triggers:
                    group = replace(group, bias=0.)
                groups[group.name] = replace(group, enabled=True, learnable=False)
                updates.append(("group-control", control.to_dict(), f"Group {group.name!r} {direction} (target/1000 tokens)", control.target_rate * 1000))
            else:
                controls.pop(key, None)
                step = command.bias_amount * (1 if command.bias_operator == "+" else -1)
                if not triggers:
                    group = replace(group, bias=group.bias + step, enabled=True, learnable=False)
                    groups[group.name] = group
                    updates.append(("bias-group", {"group": group.to_dict()}, f"Group {group.name!r} manual", group.bias))
                else:
                    for r in group.rules:
                        template = replace(r, triggers=triggers, until=until, logical_target=identity)
                        old = rules[template.key].bias if template.key in rules else 0.
                        rule = replace(template, bias=old + step)
                        rules[rule.key] = rule
                        updates.append(("bias-rule", {"rule": rule.to_dict()}, f"Group {group.name!r} scoped manual", rule.bias))
    return replace(sampling, bias_groups=tuple(groups.values()), bias_rules=tuple(rules.values()),
                   group_controls=tuple(controls.values())), updates
