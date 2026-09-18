# Reduced architecture

```text
backend logits -> sampler -> editor UI -> policy action -> EpisodeEngine
                                     |             |
                                     |             +-> token evidence
                                     +----------------> / vocabulary search

EpisodeStore <--- actions + token evidence
    |                         |
    +----> Projector          +----> Serial Policy Replay
                                      |
                                      v
                                  live edge
                                      |
                     continue / sampler / fork / SPR / end
```

## State that matters

- exact token prefix
- model/tokenizer supplied by the active backend
- sampling policy
- seed + sampler stream coordinate
- teacher policy actions

Backend evaluation state is runtime machinery, not episode state. A backend may
use incremental evaluation or complete-prefix evaluation, and the CLI can
select between those paths when supported. Neither path is evidence of its
own; the token/action ledger and Serial Policy Replay establish behavior.

The editor does not attest to private backend execution details.

## Lineage presentation

The existing episode parent fields are enough to render ancestry. Ordinary
forks form the family tree through `parent_episode_id` and `fork_boundary`.
Serial Policy Replay records retain their execution context and policy source,
but are presented separately rather than as ordinary family branches. The
lineage query and optional projector metadata are read-only views over this
ledger; they do not add another lifecycle or provenance mechanism.
