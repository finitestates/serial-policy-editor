# Dispatch: Episode-Local Online Supervised Learning

**Repository:** https://github.com/finitestates/serial-policy-editor  
**Target branch:** `policy-editor-argmax`

## Objective

Implement an experimental online supervised learning mode in Serial Policy Editor.

When the human overrides a model's proposed token, treat the intervention as a **supervised correction**. Immediately train the model on the corrected token using gradient descent, then continue the episode using the updated model weights.

The intention is to reduce the number of subsequent human interventions required to produce the desired trajectory.

**Model weights are disposable. The trajectory is the valuable artifact.**

### Important philosophical note

Yes, this is deliberately more aggressive than conventional fine-tuning.

Normally we'd worry about catastrophic forgetting, preservation of general-purpose competencies, distribution shift, KL regularization, and destabilizing a pretrained model.

Those concerns are largely irrelevant to this experiment.

Each episode starts with the original pretrained model. Weight updates need only improve performance over the remainder of that episode. Once the episode is finished, the learned weights are discarded.

If training during a ghost story destroys the model's ability to write JSON, that's acceptable. We are not optimizing a general-purpose model.

Damage to subsequent decisions **within the episode** still matters and should be measured, but don't introduce constraints solely to preserve unrelated capabilities.

This is an experiment. Unexpected behavior is useful evidence, not automatically a defect.

## 1. Learning algorithm

Start with direct supervised fine-tuning using PyTorch SGD.

For a prefix \(x_{1:t}\) and human-selected next token \(y\):

\[
\mathcal L(\theta)=-\log P_\theta(y\mid x_{1:t})
\]

Update:

\[
\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L
\]

Requirements:

- Train actual model parameters, not just a separate bias table.
- Start with full-model SGD, without momentum or weight decay.
- Use next-token cross-entropy.
- Support a configurable number of optimization steps per correction.
- Do not add KL penalties, reference-model constraints, reward models, or preference optimization.
- Keep the model in evaluation mode so dropout stays disabled, but explicitly enable autograd for supervised updates.
- Start with unquantized float32 Transformers models. Broader dtype and quantization support can come later.

For multi-token human interventions, use teacher forcing over the corrected sequence. Compute the mean cross-entropy across the corrected tokens, then perform the configured optimizer steps.

Do not train on autonomous generation or accepted proposals. Train on human overrides.

An intervention is supervision. Do not require a secondary confirmation that the model was wrong.

## 2. CLI interface

Add the following configuration:

```bash
policy-editor \
  --backend transformers \
  --model /path/to/model \
  --new-prompt "The house had been empty for fifty years." \
  --online-sgd \
  --online-lr 1e-5 \
  --online-steps 1
```

`--online-sgd` enables episode-local weight updates.

`--online-lr` controls the SGD learning rate.

`--online-steps` controls the number of gradient steps following each correction.

Learning should be disabled by default, preserving existing behavior.

Initially, support this only through the Transformers backend. If requested with llama.cpp or an unsupported model configuration, give an explicit, informative error.

Keep learning configuration distinct from `SamplerConfig`: optimization controls are not sampling controls.

## 3. Runtime interface

Implement these commands:

```text
learn steps=5
learn lr=0.0001
learn steps=20
learn steps=0
learn status
learn reset
teach last steps=20
```

Semantics:

- `learn steps=N` changes the number of SGD steps applied to future corrections.
- `learn lr=X` changes the learning rate for subsequent updates.
- `learn steps=0` disables automatic learning without ending the episode.
- `learn status` reports configuration, correction count, optimizer steps, recent losses, and current weight revision.
- `learn reset` restores the original episode-start model weights, discards optimizer state, and clears learned-update history, while retaining the already-generated token trajectory.
- `teach last steps=N` applies N additional supervised steps using the most recent human correction and its original pre-correction context, without changing the default step count.

These controls should be available during an active episode. The CLI arguments establish initial settings; runtime commands adjust them.

## 4. Integration points

Inspect the current architecture before implementing.

Relevant existing components include:

- `core/src/trajectory_editor/transformers_backend.py`
- `core/src/trajectory_editor/episode_engine.py`
- `core/src/trajectory_editor/episode_session.py`
- `core/src/trajectory_editor/core/policy_calculations.py`
- `core/src/trajectory_editor/core/actions.py`
- Existing teacher-command parsing and session-runtime modules

The Transformers backend already owns the PyTorch model. Add an optional supervised-update capability there, rather than introducing torch dependencies into the backend-neutral core interfaces.

At a successful human override:

1. Capture the exact pre-correction token prefix.
2. Resolve the intervention to the actual corrected token IDs.
3. Commit the human-selected tokens through the existing action machinery.
4. Run the configured number of supervised training steps on the captured prefix and selected tokens.
5. Invalidate any inference state computed under old model weights.
6. Recompute inference state for the current prefix.
7. Continue normal generation.

Only train on successfully committed corrections. A rejected action, diagnostic-only action, or replay handoff must not silently produce a learning update.

Pay attention to `Write`: its optimized batch-commit path deliberately avoids per-token model evaluation. It should remain possible to train on a written span using a separate teacher-forced forward pass.

The training implementation must avoid off-by-one target alignment errors.

## 5. Cache invalidation is essential

The existing Transformers backend uses `torch.inference_mode()` and maintains KV caches and last-logit snapshots.

A training update changes model parameters, invalidating cached computations even when the token prefix hasn't changed.

After any weight update:

- Invalidate the backend's KV cache and cached logits.
- Invalidate cached observations and speculative continuations.
- Invalidate or rebuild any retained beam/batch cache state that could depend on previous weights.
- Re-evaluate the committed prefix using the updated parameters.

Consider introducing a monotonically increasing `weight_revision` and incorporating it into appropriate cache-validity checks.

A matching token prefix alone is not sufficient to establish cache validity after learning.

Ensure training creates fresh autograd-enabled tensors and does not accidentally reuse tensors created under inference mode.

Also account for training failures: never silently continue with stale inference state or a partially invalid model.

## 6. Episode lifecycle and time travel

Learning state belongs to an episode, not to the pretrained checkpoint.

A new episode must start with original model weights. Ending or discarding the current episode must not contaminate subsequent episodes.

Rewind, fork, replay, and branch switching require special consideration because existing branches share an inference backend.

Ultimately, learning updates should be treated as episode events with sufficient information to reconstruct the learned state associated with a branch.

A suitable event representation might include:

```python
LearningEvent(
    prefix_token_ids=...,
    corrected_token_ids=...,
    learning_rate=...,
    steps=...,
    weight_revision=...,
)
```

The token-level correction and its training effect must remain causally associated.

**For the first implementation, prioritize a correct linear-episode prototype.** If rewind, fork, replay, beam, or branch switching cannot yet preserve the correct learned model state, explicitly disable the incompatible operation while online learning is active, with a clear explanation.

Do not silently allow old and new model weights to become mixed across branches. Do not break existing behavior when online learning is disabled.

Once the basic learner works, extend the event/replay machinery to support branch-local learned state. Restoring the base checkpoint and deterministically replaying learning events is an acceptable initial strategy, even if slow.

## 7. Experimental observability

Record useful information about each learning event:

- Decision boundary and selected token(s)
- Original model proposal
- Loss before training
- Loss after training
- Learning rate and SGD step count
- Gradient magnitude
- Weight revision

The immediate loss reduction is a sanity check, not the ultimate success metric.

The actual question is:

**Does online learning reduce the number of human corrections needed over subsequent portions of the same episode?**

Avoid conflating improved training-example loss with improved future behavior.

Make it easy to compare ordinary frozen generation against the online-learning mode.

## 8. Tests and validation

Implement focused tests using a tiny, real trainable causal language model rather than relying exclusively on fake logits.

Minimum coverage:

1. A correction produces gradients and changes model weights.
2. A suitable SGD step increases the corrected token's likelihood at the original prefix.
3. Multi-token supervision is aligned correctly.
4. `--online-steps N` and runtime `learn steps=N` apply the requested number of updates.
5. `teach last steps=N` retrains the correct example without changing future defaults.
6. `learn reset` restores original model behavior at the same prefix.
7. Cached inference cannot survive a weight change incorrectly.
8. Separate episodes cannot inherit each other's learned weights.
9. Unsupported backends/configurations fail explicitly.
10. Existing non-learning runtime, replay, and argmax contracts continue passing.

Do not claim cross-backend or real-model correctness based exclusively on mocked tests.

For the first experimental mode, prefer deterministic plain argmax with ordinary policy settings and no added noise. The sampler itself should remain unchanged.

## 9. Scope and implementation priorities

Prioritize in this order:

**Phase A — Working learning loop**

Implement direct SGD on human overrides, runtime learning configuration, reset, cache invalidation, and basic metrics. Support one linear episode correctly.

**Phase B — Experimental controls**

Implement `teach last`, useful training diagnostics, and tests comparing predictions before and after updates.

**Phase C — Rewind, fork, and replay**

Make learned states correctly reconstructible and branch-local, preserving the project's existing episode semantics.

Avoid building an elaborate generalized training framework. This should be a small, understandable experimental feature.

Do not add LoRA, fast-weight architectures, replay buffers, or elaborate regularization unless the direct SGD experiment demonstrates a need.

## Definition of done

I should be able to launch an episode with `--online-sgd`, override a model token, watch supervised training occur, and observe that subsequent predictions come from the modified model.

I should be able to increase the number of SGD steps live, apply additional training to the last correction, inspect basic learning diagnostics, and reset the model to its original weights.

Normal operation with learning disabled must remain unchanged.

If advanced episode operations aren't supported in the initial cut, they must fail explicitly rather than producing incorrect replay or cache behavior.

**The central research question is whether direct human-supervised backpropagation, applied at individual generation errors, helps a model finish one trajectory with progressively less intervention.**

The purpose is to run that experiment, not to engineer away its potentially strange results.

Please inspect the existing code and contracts first, implement the smallest correct vertical slice, run the relevant tests, and report what works, what remains unsupported, and any observed model behavior. Do not treat successful unit tests as proof that the learning scheme improves actual trajectories.
