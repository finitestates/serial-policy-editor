"""Shared policy diagnostics and responsive candidate-table columns."""

from dataclasses import dataclass

from .domain import Candidate


@dataclass(frozen=True)
class CandidateColumns:
    policy: bool = False
    logit_view: str = "none"
    width: int | None = None

    def _fits(self, threshold: int) -> bool:
        return self.width is None or self.width >= threshold

    @property
    def columns(self) -> tuple[tuple[str, int], ...]:
        columns = []
        if self.logit_view in {"raw", "all"}:
            columns.append(('raw-logit', 10))
        if self.logit_view in {"effective", "all"}:
            columns.append(('eff-logit', 10))
        if self.logit_view in {"delta", "all"}:
            columns.append(('Δlogit', 9))
        if self.policy:
            columns.append(('Δrank', 6))
            if self.logit_view not in {"delta", "all"}:
                columns.append(('Δlogit', 8))
            if self._fits(92):
                columns.append(('pol-rank', 8))
        if not self.policy or self._fits(68):
            columns.append(('raw-p', 8))
        if self.policy and self._fits(114):
            columns.append(('pol-p', 8))
        if not self.policy or self._fits(50):
            columns.append(('decode-p', 8))
        if self._fits(134 if self.policy else 78):
            columns.append(('token-id', 8))
        return tuple(columns)

    @property
    def heading(self) -> str:
        return ''.join(f'  {label:>{width}}' for label, width in self.columns)

    def values(self, candidate: Candidate) -> str:
        def value(label: str) -> str:
            if label == 'raw-logit':
                return (f'{candidate.raw_logit:+.3f}'
                        if candidate.raw_logit is not None else '--')
            if label == 'eff-logit':
                return (f'{candidate.effective_logit:+.3f}'
                        if candidate.effective_logit is not None else '--')
            if label == 'Δlogit':
                return (f'{candidate.policy_logit_adjustment:+.3f}'
                        if candidate.policy_logit_adjustment is not None else '--')
            if label == 'Δrank':
                return (f'{candidate.rank - candidate.policy_rank:+d}'
                        if candidate.policy_rank is not None else '--')
            if label == 'pol-rank':
                return str(candidate.policy_rank) if candidate.policy_rank is not None else '--'
            if label == 'token-id':
                return str(candidate.token_id)
            probability = {
                'raw-p': candidate.model_probability,
                'pol-p': candidate.policy_probability,
                'decode-p': candidate.decoder_probability,
            }[label]
            return f'{probability:.2%}' if probability is not None and probability > 0 else '--'

        return ''.join(f'  {value(label):>{width}}' for label, width in self.columns)
