"""Compact learning notices with an on-demand explanation of the latest event.

Reports are presentation state only: no observations, weights, or actions change.
"""
from __future__ import annotations

import math


def _norm(values):
    return math.sqrt(sum(float(v) ** 2 for v in values))


def _token(token_id, token_text):
    if token_id is None:
        return 'unknown'
    if token_text is None:
        return f'token {token_id}'
    text = repr(token_text(token_id))
    return f'{text[:45] + "…" if len(text) > 48 else text} (id {token_id})'


def _reason(result):
    if not result.enabled:
        return 'learner disabled'
    if result.learning_gate == 'sampler':
        return ('no evidence: already eligible' if result.sampler_eligible else
                'evidence admitted: excluded from sampler')
    if result.severity == 0:
        return f'no evidence: rank inside dead zone 1–{result.dead_zone_rank}'
    return f'evidence admitted: severity {result.severity:.3g}'


def _probability(value):
    return 'unavailable' if value is None else f'{value:.4%}'


def _choice_details(result, token_text):
    return [
        f'Chosen: {_token(result.chosen_token_id, token_text)}',
        f'Proposal: {_token(result.proposal_token_id, token_text)} — '
        + ('teacher chose differently' if result.proposal_rejected else 'teacher matched proposal'),
        f'Before update: policy rank {result.old_policy_rank}; '
        f'policy probability {_probability(result.old_policy_probability)}; '
        f'sampler probability {_probability(result.sampler_probability)}',
        f'Gate ({result.learning_gate}): {_reason(result)}.',
        f'Rejection strength {result.rejection_strength:g}; target {result.rejection_target}'
        + (' (no rejection term on agreement).' if not result.proposal_rejected else '.'),
    ]


def _channel_details(result, *, fast=False):
    prefix = 'fast_' if fast else ''
    old = result.old_fast_z if fast else result.old_z
    new = (result.new_fast_z if fast else result.new_z) or (0.,) * len(old)
    evidence = getattr(result, prefix + 'learning_evidence')
    step = getattr(result, prefix + 'learning_delta')
    decay = result.effective_fast_decay if fast else result.effective_decay
    configured_decay = result.fast_decay if fast else result.decay
    step_norm = getattr(result, prefix + 'learning_step_norm')
    raw_norm = _norm(evidence)
    clipped_step = not math.isclose(raw_norm, step_norm, rel_tol=1e-7, abs_tol=1e-10)
    # Reconstruct the state immediately before memory bounds were applied.
    bounded = _norm([(1-decay)*a + b - c for a, b, c in zip(old, step, new)])
    strength = result.sampling.latent_fast_strength if fast else result.sampling.latent_strength
    norm = getattr(result, prefix + 'z_norm')
    return [
        f'{"Fast" if fast else "Slow"} memory ({len(old)} dimensions):',
        f'  New learning: {step_norm:.4g} (evidence {raw_norm:.4g}; '
        f'step clipping {"applied" if clipped_step else "not needed"}).',
        f'  Decay: {getattr(result, prefix + "decay_norm"):.4g} removed; '
        f'rate {decay:g} applied / {configured_decay:g} configured; trigger {result.decay_on}.',
        f'  Memory bounds: {bounded:.4g} adjustment.' if bounded > 1e-10 else '  Memory bounds: no adjustment.',
        f'  Net movement: {getattr(result, prefix + "update_norm"):.4g}.',
        f'  Memory size (z norm): {_norm(old):.4g} → {norm:.4g}.',
        f'  Token bias bound: ±{abs(strength)*norm:.4g} logits at strength {strength:g}.',
    ]


def _latent_details(result):
    lines = _channel_details(result)
    if result.old_fast_z:
        lines.extend(_channel_details(result, fast=True))
    elif result.sampling.latent_preference_fast_z:
        lines.append('Saved fast memory also steers output; fast learning is off.')
    if result.learning_scheme == 'fisher-kl-v2':
        lines.extend([
            'v2 exact policy-KL update:',
            f'  Gradient norm: {result.raw_gradient_norm:.4g}; Fisher mode: {result.fisher_mode}; '
            f'condition estimate: {result.fisher_condition_estimate:.4g}.',
            f'  Requested learning KL: {result.requested_learning_kl:.4g}; '
            f'predicted Fisher KL: {result.predicted_fisher_kl:.4g}; '
            f'exact learning KL: {result.exact_learning_kl:.4g} '
            f'({result.kl_line_search_iterations} line-search iterations).',
            *([
                f'  Fast learning KL budget: {result.fast_requested_learning_kl:.4g}; '
                f'predicted Fisher KL: {result.fast_predicted_fisher_kl:.4g}; '
                f'exact fast learning KL: {result.fast_exact_learning_kl:.4g} '
                f'({result.fast_kl_line_search_iterations} line-search iterations).',
            ] if result.fast_requested_learning_kl > 0.0 or result.fast_exact_learning_kl > 0.0 else []),
            f'  Pairwise margin: {"unavailable" if result.pairwise_margin is None else f"{result.pairwise_margin:.4g}"}; '
            f'pairwise loss: {result.pairwise_loss:.4g}; '
            f'rejection gradient norm: {result.rejection_gradient_norm:.4g}.',
            f'  Safety clips: step {"yes" if result.step_clipped else "no"}; '
            f'norm {"yes" if result.norm_clipped else "no"}.',
            f'  Influence: mode {result.sampling.latent_influence_mode}; '
            f'target {result.sampling.latent_influence_kl:.4g}; '
            f'deployment KL {result.latent_deployment_kl:.4g}; '
            f'global gain {result.latent_effective_gain:.4g}; '
            f'user multiplier {result.latent_user_multiplier:.4g}; '
            f'gain capped {"yes" if result.latent_gain_capped else "no"}.',
            f'  Raw contribution RMS: slow {result.latent_slow_raw_rms:.4g}; '
            f'fast {result.latent_fast_raw_rms:.4g}; combined {result.latent_combined_raw_rms:.4g}; '
            f'logit RMS {result.latent_effective_logit_rms:.4g}; '
            f'top-N range {result.latent_top_logit_min:.4g} .. {result.latent_top_logit_max:.4g}.',
        ])
    lines.extend([
        'New learning and decay are vector movements; their sizes do not simply add.',
        'Z norm is memory magnitude, not confidence or the number of preferences learned.',
        'The bias bound is a per-channel upper bound, not a measured token probability change.',
    ])
    return lines


def _group_details(result):
    lines = [f'Decay: rate {result.effective_decay:g} applied / {result.decay:g} configured; '
             f'trigger {result.decay_on}.', f'Net movement across groups: {result.update_norm:.4g}.']
    if not result.old_group_weights:
        lines.append('No manual groups available to fit.')
    for name, old in result.old_group_weights.items():
        new = result.new_group_weights[name]
        skipped = (result.skipped or {}).get(name)
        if skipped:
            lines.append(f'  {name}: {old:.4g} → {new:.4g}; skipped: {skipped}.')
            continue
        evidence = (result.evidence or {}).get(name, 0.)
        decay_delta = -result.effective_decay * old
        adjustment = new - (old + evidence + decay_delta)
        lines.append(f'  {name}: {old:.4g} → {new:.4g}; evidence {evidence:+.4g}; '
                     f'decay change {decay_delta:+.4g}; clipping/bounds change {adjustment:+.4g}.')
    return lines


def _movement(result, latent):
    if result.update_norm == 0 and (not latent or result.fast_update_norm == 0):
        return 'unchanged'
    if latent:
        learning = result.learning_step_norm + result.fast_learning_step_norm
        decay = result.decay_norm + result.fast_decay_norm
        if not learning and not decay:
            return 'memory bounds changed state'
        return 'learned + decayed' if learning and decay else ('learned' if learning else 'decayed')
    return f'net {result.update_norm:.3g}'


def _publish(io, key, title, sections, summary):
    previous = getattr(io, '_learning_readout', None)
    combined = dict(previous[2]) if previous and previous[0] == key else {}
    combined.update(sections)
    io._learning_readout = (key, title, combined)
    io.write(summary)


def selection_notice(io, result, *, latent, token_text=None, episode_id=None):
    label = 'Latent' if latent else 'Groups'
    reason = _reason(result)
    if result.enabled and result.severity > 0:
        reason = 'sampler excluded' if result.learning_gate == 'sampler' else f'severity {result.severity:.3g}'
    if result.learning_gate == 'rank' and result.severity == 0 and result.enabled:
        reason = f'no evidence: rank ≤{result.dead_zone_rank}'
    if not latent and result.enabled and not result.evidence:
        reason = 'no eligible groups'
    details = _choice_details(result, token_text)
    details.extend(_latent_details(result) if latent else _group_details(result))
    summary = f'{label} [learning]: {reason} · {_movement(result, latent)}'
    if latent:
        summary += f' · z {result.z_norm:.4g}'
        if result.old_fast_z:
            summary += ' (slow)'
    boundary = result.observation_boundary + 1
    _publish(io, (episode_id, 'selection', boundary), f'Teaching selection @ boundary {boundary}',
             {label: details}, summary)


def write_notice(io, result, *, token_text=None, episode_id=None):
    sections = {}
    summaries = []
    for label, aggregate, tokens in (
        ('Groups', result.group_result, result.group_token_results),
        ('Latent', result.latent_result, result.latent_token_results),
    ):
        if aggregate is None:
            continue
        count = aggregate.write_evidence_tokens
        count = sum(t.severity > 0 for t in tokens) if count is None else count
        lines = [f'Rejection strength {aggregate.rejection_strength:g}; target {aggregate.rejection_target}.',
                 f'{count}/{result.token_count} tokens supplied gate-admitted evidence; '
                 f'{result.token_count-count} gate skips.',
                 f'Write {aggregate.write_reduction}: {count} evidence tokens, '
                 f'scale {aggregate.write_evidence_scale:.4g}; decay evaluated once.']
        if tokens:
            matches = sum(not t.proposal_rejected for t in tokens)
            lines.append(f'Proposal agreement: {matches}/{len(tokens)} tokens (separate from gate admission).')
            lines.append('Per-token evidence, measured after the preceding written tokens:')
            for t in tokens:
                lines.append(f'  {_token(t.chosen_token_id, token_text)}: rank {t.old_policy_rank}; '
                             f'{"different choice" if t.proposal_rejected else "matched proposal"}; {_reason(t)}.')
        lines.extend(_latent_details(aggregate) if label == 'Latent' else _group_details(aggregate))
        sections[label] = lines
        summaries.append(f'{label.lower()} {count}/{result.token_count} evidence, '
                         f'{_movement(aggregate, label == "Latent")}')
    _publish(io, (episode_id, 'write', result.boundary_after),
             f'Teaching Write @ boundaries {result.boundary_before}–{result.boundary_after}',
             sections, 'Write [learning]: ' + ' · '.join(summaries))


def show_learning_details(io, *, episode_id=None):
    report = getattr(io, '_learning_readout', None)
    lines = ['Learning details', '']
    if report is None or report[0][0] != episode_id:
        lines.append('No teaching report for this episode in this session yet.')
    else:
        _, title, sections = report
        lines.extend([title + ' (last teaching event in this session)', ''])
        for label, details in sections.items():
            lines.extend([label, *details, ''])
    lines.extend([
        'Hold, the explicit Accept command, and replay do not teach or decay memory.',
        'Selecting the proposal by rank/Enter is a teaching action; the gate may skip its evidence.',
        'This report is historical; later holds, manual edits, or rewinds do not recompute it.',
    ])
    page = getattr(io, 'page', io.write)
    page('\n'.join(lines))
