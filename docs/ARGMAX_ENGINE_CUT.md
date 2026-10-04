# Argmax engine cut

This variation selects the maximum score after explicit policy adjustments and
optional perturbation. Full model logits remain available for search.

## Removal inventory

| Surface | Decision |
| --- | --- |
| Categorical draw / cumulative probability lookup | Remove |
| Top-p cumulative mass filter | Remove |
| Typical-p entropy ordering and cumulative mass | Remove |
| Tail-free probability derivatives and cumulative mass | Remove |
| Conditional Gumbel top-k exponential race | Remove |
| Stochastic beam / gbeam without replacement | Remove, including command and UI paths |
| Min-p relative probability cutoff | Retain as `score >= maximum + log(min_p)` |
| Eager candidate softmax during proposal construction | Replace with lazy diagnostic probabilities |
| Model / policy / eligible-set probability overlays | Retain as optional diagnostics, not selection rules |
| Deterministic beam cumulative log-probability scoring | Retain; it uses no CDF |
| Gaussian / Logistic / Laplace / Student-t / Uniform / Gumbel noise | Retain |
| Top-k eligibility and selective-noise-k | Retain independently; eligibility defaults to none |

Plain argmax is the default kernel. Gumbel remains the categorical-equivalent
control. Noise is an opt-in candidate overlay; the default rank / token ID / text
view stays unchanged. Noise is the additive change in eligible score, including
zero for eligible candidates left untouched; excluded candidates show no value.

## Deliberate compatibility break

Removed filter fields and the categorical kernel are rejected in saved sampler
records and CLI/profile inputs. Old tapes/workspaces with those settings need an
explicit conversion decision; removed settings are not silently discarded.
Tests, the independent reference kernel, benchmarks, and historical documentation
may describe the previous engine. They are not rewritten to conceal that break.
No tests are run for this cut. Runtime, replay, and terminal behavior remain
unverified until the next review/validation pass.

## Overlay navigation follow-up

Direct toggles are `l` (raw logits), `L` (diff from raw argmax), `~` (noise),
and `%` (probability diagnostics). `C` resets all columns, including policy
columns, to rank / token ID / text. Neighbor margin, z-score, and the `c` column
cycle are removed. Context aliases `c N` / `c all` remain.

`columns` reports the active overlays; `columns logit diff noise` selects an
exact set. `overlay NAME` toggles, and `overlay NAME on|off` explicitly sets one.
All routes share the same overlay selection so toggling a shortcut after a
named command reliably turns that same overlay off. Presentation changes are
session preferences and do not alter replay or proposal selection.

No tests were run for this follow-up. Source syntax and diff review are the only
checks performed; terminal behavior remains unverified.
