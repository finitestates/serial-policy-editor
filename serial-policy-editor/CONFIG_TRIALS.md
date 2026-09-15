# Steering configs to try — 0.4.0

Based on 0.4.0 plus the live group-learning toggle on `main`. These are hypotheses
to test, not measured quality claims. **My first picks: 00 → 05 → 06 → 09 → 11**,
then the group-only trials. Commands have been checked against the CLI parser
and numeric configuration validation; quality still needs real-model trials.

The main distinction: appearance objectives keep working during `h` holds;
teacher learners update on live rank selections, plus typed writes only when
enabled. Online learning fits named manual groups; latent learning spreads
preferences through model token features, so its effects can reach words you
never put in a group.

## One-time setup

Use Bash with your current 0.4.0 installation activated. Run from the
`serial-policy-editor` directory. Replace the model path and prompt with yours;
keep them the same across comparisons. The examples assume llama.cpp. For
Transformers, change `BACKEND` and use your local model directory.

```bash
MODEL='/absolute/path/to/your/model.gguf'
BACKEND='llama.cpp'
PROMPT='At dusk, the keeper returned to the abandoned lighthouse. Inside,'
mkdir -p config-trials

# Optional: put your usual hardware flags here, unchanged across trials.
HARDWARE=()  # Example: HARDWARE=(--n-gpu-layers 99)

# Shared sampler: 73 is an arbitrary fixed random seed;
# 0.8 temperature, top 40 tokens, 95% cumulative probability,
# and a 5%-of-best probability floor. These are the normal sampler defaults.
# Repeat penalty 1 means no repetition penalty.
# 512 is a checkpoint allowance, NOT automatic generation or termination.
# 32 is the default hold length; explicit `h N` overrides it.
trial() {
  local name="$1"
  shift
  policy-editor --backend "$BACKEND" --model "$MODEL" \
    "${HARDWARE[@]}" \
    --workspace "config-trials/${name}.sqlite3" \
    --new-prompt "$PROMPT" \
    --seed 73 --temperature 0.8 --top-k 40 --top-p 0.95 --min-p 0.05 \
    --repeat-penalty 1 --max-tokens 512 --hold-default 32 --policy-view \
    "$@"
}

cat > config-trials/groups.yaml <<'YAML'
groups:
  atmosphere:
    - shadow
    - silhouette
    - mist
    - silence
    - gathering storm
  motion:
    - turn
    - step
    - reach
    - cross
    - open
YAML
```

Each invocation starts a fresh episode with empty learner memory unless it
explicitly loads a preset. Reusing a trial name adds an episode to that workspace;
it does not inherit the previous episode's learning. Use a new name for repeats
if you want one episode per file.

## 00 — Unsteered control

```bash
trial 00-control
```

**Expect:** the reference point for wording, repetition, and how often you feel
the need to correct the model. Use the same teaching pattern as later trials,
even though this one doesn't learn from it.

## 01 — Gentle appearance control

```bash
# 0.5 requests sqrt(2), about 1.41 times the captured baseline odds.
trial 01-gentle-group --groups config-trials/groups.yaml --group-level 0.5
```

At the teacher prompt, enter these separately:

```text
b atmosphere +
h 256
b
```

**Expect:** more atmosphere vocabulary with relatively little disruption. Watch
completed appearances and controller pressure in `b`, not just individual token
ranks. A phrase entering the candidate list is not the same as completing it.

## 02 — Strong appearance control

```bash
# 2 requests 2^2 = 4 times baseline odds. It does NOT add +2 logits.
trial 02-strong-group --groups config-trials/groups.yaml --group-level 2
```

Use the same teacher commands as 01. **Expect:** a more obvious motif, with more
risk of repetitive or awkward word choices. Comparing 01 and 02 tests whether
the controller feels responsive without needing any teacher learning.

Two useful direction checks, each in its own fresh run:

```bash
trial 02-suppress --groups config-trials/groups.yaml --group-level 2
# Inside SPE: b atmosphere -     # target one-quarter of baseline odds

trial 02-maintain --groups config-trials/groups.yaml --group-level 1
# Inside SPE: h 128              # let a pattern develop
# Then:      b atmosphere =     # capture and maintain the current rate
# Then:      h 256
```

Maintain ignores the level multiplier. It is most informative after some text
exists. All these are soft targets, not frequency guarantees. A rare group can
remain rare even at four times its baseline odds. Bare `=` means maintain;
`off` clears the objective. An explicit `+0.5` instead creates a manual bias.

## Setup for online group learning (03, 04, and 11 only)

No preset conversion is needed. Launch the recipe with `--groups` and
`--online-learning`, then opt in at the teacher prompt before selecting a token:

```text
b atmosphere learn on
b motion learn on
b
```

These commands load the named definitions from your groups YAML directly.
For 11, enable only `motion`; `atmosphere` will use an appearance objective.
`--learnable-groups` only restricts the eligible set; the live commands opt it in.
`b GROUP learn off` freezes the current learned amount without clearing it.
The toggle is saved with the episode and exported presets. Membership and numeric
edits preserve the choice; appearance activation switches learning off.

## 03 — Conservative named-group learner

```bash
# 0.05 = learning rate; 0.25 = maximum change per group per teaching action.
# -4 / +4 = bounds on each learned group bias.
# 1000 = rank-severity scale; 1 = policy-rank dead zone.
# 0 rejection = no extra push against the rejected proposal.
# 0 decay = keep old weights between teaching actions.
trial 03-online-default --groups config-trials/groups.yaml \
  --online-learning --learnable-groups atmosphere motion \
  --learning-rate 0.05 --learning-max-step 0.25 \
  --learning-min-bias -4 --learning-max-bias 4 \
  --learning-severity-cap 1000 --learning-dead-zone-rank 1 \
  --learning-rejection-strength 0 --learning-decay 0
```

**Expect:** slow, interpretable movement in the two group amounts. Repeatedly
choosing atmosphere tokens should tend to increase atmosphere bias; choosing
outside it can reduce it. It learns token-route preferences, not the completed
appearance-rate target used by 01/02. Inspect amounts with `b`.

After launch, enter `b atmosphere learn on` and `b motion learn on` before
teaching. Keep numeric steering edits out of the comparison so the changes you
see come from learning. Appearance objectives exclude their group from the
online fitter. Reading `b` is fine.

## 04 — Responsive named-group learner

```bash
# Change only severity relative to 03: every choice outside policy rank 1
# receives full severity, while the same rate, step cap, and bounds remain.
trial 04-online-full --groups config-trials/groups.yaml \
  --online-learning --learnable-groups atmosphere motion \
  --learning-rate 0.05 --learning-max-step 0.25 \
  --learning-min-bias -4 --learning-max-bias 4 \
  --learning-dead-zone-rank 1 --learning-no-severity-attenuation \
  --learning-rejection-strength 0 --learning-decay 0
```

Use the same two `learn on` commands as 03 before teaching.

**Expect:** group amounts move noticeably sooner when you make modest rank
corrections. If 03 feels inert and this works, severity attenuation is a better
first suspect than the step limit. Watch for one group dominating after a few
choices.

## Shared latent settings for 05–11

```bash
LATENT=(
  --latent-preference
  --latent-dimension 64          # length of preference memory / token features
  --latent-seed 9137             # fixed feature-projection seed, separate from sampler seed
  --latent-learning-rate 0.05    # amount of new preference evidence
  --latent-strength 1           # how strongly stored memory changes logits
  --latent-max-step 0.25         # cap on one slow-memory learning step's vector length
  --latent-max-norm 4            # cap on total slow-memory vector length
  --latent-decay 0               # fraction of old slow memory forgotten per learning update
  --latent-dead-zone-rank 1      # no new learning evidence for policy rank 1
  --latent-severity-cap 1000     # scale at which rank correction reaches full severity
  --latent-rejection-strength 0 # no extra negative-proposal term
)
```

Later flags override earlier scalar values, so each recipe below shows its
changes to this shared baseline. Latent step/norm limits are vector lengths,
not per-token logit limits. Strength changes influence, learning rate changes
acquisition; they aren't interchangeable in a running policy.

## 05 — Conservative latent learner

```bash
trial 05-latent-default "${LATENT[@]}"
```

**Expect:** subtle, accumulating effects after repeated deliberate choices.
This is the baseline for all the following latent comparisons. Generalization
can be useful or surprising: token-feature similarity does not guarantee the
specific stylistic similarity you had in mind.

## 06 — Make ordinary corrections count more

```bash
# Only lower the severity scale from 1000 to 50.
trial 06-latent-sensitive "${LATENT[@]}" --latent-severity-cap 50
```

**Expect:** faster response to normal candidate-list corrections, without making
every correction equally strong. With dead zone 1, policy rank 10 receives
about 0.59 severity here versus 0.33 in 05; policy rank 51 gets full severity
versus about 0.57. This is my first adjustment if the default feels too quiet.

## 07 — Full-strength corrections

```bash
# Only remove severity attenuation. Policy rank 1 still produces no evidence.
trial 07-latent-full "${LATENT[@]}" --latent-no-severity-attenuation
```

**Expect:** the strongest response of 05/06/07 to near-top choices. Useful for
finding out whether the learner can register small but deliberate preferences.
It may also overvalue inconsequential edits. Full severity is still subject to
the 0.25 step and 4 total-memory limits.

## 08 — Learn what I rejected, too

```bash
# Compare with 06. 0.5 adds half-strength negative-proposal evidence
# when your chosen token differs from the sampler's proposal.
trial 08-latent-rejection "${LATENT[@]}" \
  --latent-severity-cap 50 --latent-rejection-strength 0.5
```

**Expect:** more decisive movement away from recurring unwanted proposals.
The extra term pushes against the rejected token's features relative to the
policy mean; it is not a 50% probability penalty. Because the proposal is a
sample, this can also punish a perfectly acceptable alternative. Compare how
often you correct the same tendency again, and whether coherence suffers.

## 09 — Stable taste plus short-term adaptation

```bash
# Compare with 06. Keep its slow channel and add a fast channel:
# 0.20 = four times the slow learning rate; 0.10 = forget 10% per update;
# 0.5 = fast memory's logit strength; 0.25 / 1 = fast step / total norm caps.
trial 09-latent-fast-slow "${LATENT[@]}" \
  --latent-severity-cap 50 --latent-fast-slow \
  --latent-fast-learning-rate 0.20 --latent-fast-decay 0.10 \
  --latent-fast-strength 0.5 --latent-fast-max-step 0.25 \
  --latent-fast-max-norm 1
```

**Expect:** recent preferences take effect sooner while slower accumulated taste
persists. Try switching from atmospheric description to physical action midway
through your teaching. Without reinforcing evidence, the fast channel retains
roughly half its old contribution after 7 learning updates. **It does not decay
just because you generated 7 tokens, ran a hold, or waited.**

## 10 — Teach with short typed spans

```bash
# Exactly 09, plus permission to learn from live t/x writes.
trial 10-latent-writes "${LATENT[@]}" \
  --latent-severity-cap 50 --latent-fast-slow \
  --latent-fast-learning-rate 0.20 --latent-fast-decay 0.10 \
  --latent-fast-strength 0.5 --latent-fast-max-step 0.25 \
  --latent-fast-max-norm 1 --learn-from-write
```

**Expect:** your inserted prose can teach local wording preferences without
selecting every token by rank. Try short, intentional clauses. Evidence from a
whole write is summed, then clipped and decayed once: a long paste is not a
proportionally larger guaranteed lesson and can hit the step cap. For a fair
09/10 comparison, type the same spans in both; only 10 learns from them.

## 11 — Combined: atmosphere objective + motion fitter + latent memory

```bash
# Deliberately give each system a different role.
# Atmosphere: adaptive 2x baseline-odds objective, activated below.
# Motion: online manual fitter, rate 0.05, step 0.25, bounds +/-4,
#         full severity outside policy rank 1, no rejection or decay.
# Latent: recipe 09, for preferences beyond these named groups.
trial 11-combined --groups config-trials/groups.yaml \
  --group-level 1 \
  --online-learning --learnable-groups motion \
  --learning-rate 0.05 --learning-max-step 0.25 \
  --learning-min-bias -4 --learning-max-bias 4 \
  --learning-dead-zone-rank 1 --learning-no-severity-attenuation \
  --learning-rejection-strength 0 --learning-decay 0 \
  "${LATENT[@]}" --latent-severity-cap 50 --latent-fast-slow \
  --latent-fast-learning-rate 0.20 --latent-fast-decay 0.10 \
  --latent-fast-strength 0.5 --latent-fast-max-step 0.25 \
  --latent-fast-max-norm 1
```

Inside SPE, enter `b atmosphere +` and `b motion learn on`, then teach through
rank selections. Don't activate a motion appearance objective: that would exclude motion from
the fitter. **Expect:** atmosphere survives unattended holds, motion learns your
choice of action words, and latent memory generalizes your other preferences.
This is the integration trial after the isolated ones, with the greatest chance
of systems amplifying or counteracting one another. It isn't a clean one-dial
comparison with 09. If it feels oversteered, first repeat it with
`--latent-strength 0.5 --latent-fast-strength 0.25` to test reduced latent influence.

## 12 — Bonus: relative lexical taste, without learning

```bash
cat > config-trials/reference.yaml <<'YAML'
shadow: 10
silhouette: 3
outline: 1
YAML

# 10:3:1 are relative lexical weights, not promised frequencies.
# 0.25 is the default overall reference influence.
trial 12-reference-light --reference config-trials/reference.yaml \
  --reference-strength 0.25

# Change only overall influence, to four times the previous strength.
trial 12-reference-strong --reference config-trials/reference.yaml \
  --reference-strength 1
```

**Expect:** lexical preference without teacher corrections or group activation.
The stronger version should make the difference easier to notice where the
compiled lexical branches compete, but doesn't promise 10 shadows per outline.
If the isolated result is useful, add the light reference to 11 in a new run.

## 13 — Learn from choices outside the sampler's candidate set

```bash
# Your fixed-rank gate: ignore policy ranks 1–5, learn fully beyond them,
# with a chosen-versus-proposed feature direction when they differ.
trial 13-rank-gate "${LATENT[@]}" \
  --latent-dead-zone-rank 5 --latent-no-severity-attenuation \
  --latent-rejection-strength 1

# Replace that cutoff with actual sampler eligibility. The other launch
# settings, including sampler seed and latent memory limits, stay the same.
trial 13-sampler-gate "${LATENT[@]}" \
  --latent-learning-gate sampler --latent-rejection-strength 1
```

**Expect:** the second run learns from tokens your sampler could not emit, while
ignoring new evidence from alternatives it already considers. Rank 15 may survive
in a broad distribution, while rank 4 may be excluded in a concentrated one.
Sampler mode gives excluded choices full severity and ignores rank severity flags.
It retains the existing update direction; it does not stop an individual step
at the exact admission threshold or promise admission after one correction.

For group learning, add `--learning-gate sampler` to 03 or 04 and use their same
`b GROUP learn on` commands. Add `--learn-from-write` to compare typed-span learning;
coincidental agreement remains supported and eligible tokens supply zero evidence.

Readouts distinguish already-eligible choices from excluded choices. **Configured
decay still applies**, including to fast memory, so a skipped correction can still
show memory movement. These examples keep decay at the shared baseline of zero.
To test only the new gate against another configuration, retain that configuration's
decay, strength, rate, and rejection settings. Leave sampler filters enabled:
with the entire vocabulary eligible, this mode supplies no new evidence.

## A small protocol that should make your report useful

1. For group trials, activate at the same boundary and generate at least 256
   tokens. The default controller window is 256 tokens; a few words are noisy.
2. For learner trials, aim for 20–30 deliberate rank selections with short
   `h 16` gaps, then an `h 128` stretch to hear the learned result. Holds don't
   teach. Use comparable preferences, not blindly identical raw-rank numbers:
   after divergence those numbers can name different words.
3. Use the same prompt and sampler seed for the first pass. Repeat favorites
   with `--seed 74` and `--seed 75` appended to their `trial` commands. Keep
   `--latent-seed 9137` fixed so you aren't changing two random systems at once.
4. Report the trial name, approximately how many corrections it needed, whether
   your preferred style survived the long hold, unwanted repetition/wording,
   and whether teaching felt slower. For group trials, include `b` diagnostics;
   for latent trials, note whether memory norms plateaued near their caps.

Rank severity uses the **adjusted policy rank**, even though you choose tokens
by raw rank. With dead zone `d` and severity scale `c`, it is zero at ranks
`r <= d`; otherwise it is `min(1, log(1+r-d) / log(1+c))`. With full severity
enabled, it is simply 1 outside the dead zone. Raising the dead zone to 5 is a
useful follow-up if you want the learner to ignore evidence from policy ranks
1–5; decay can still forget existing memory on those learning updates.

Don't use replay to compare *learning acquisition*: replay restores saved
learning state rather than training the teacher learners again. Fresh episodes
are the straightforward comparison here. A preset carries learned policy state,
but isn't a complete record of your learner launch settings; retain the command.

To save a favorite result, find its episode number with `--list`, then export:

```bash
policy-editor --workspace config-trials/09-latent-fast-slow.sqlite3 --list
# Replace '#1' below with the episode you actually want.
policy-editor --workspace config-trials/09-latent-fast-slow.sqlite3 \
  --project '#1' --biases-only > config-trials/favorite-biases.json
```

### Dials I'd leave alone on the first pass

- `--learning-epsilon`: finite-difference probe size for fallback paths, not
  learning strength. Normal fixed-group fitting uses analytical gradients.
- `--latent-projection-chunk-size`: construction memory/speed tradeoff, not
  intended preference strength.
- Latent dimension and projection seed: useful later for robustness, but they
  change the feature representation. Changing projection seed resets saved
  preference memory; compare fresh runs. More dimensions aren't automatically
  stronger learning.
- Group window, pressure cap, and tolerance: these exist in saved control
  records but don't have corresponding ordinary launch flags. The default
  window is 256, pressure cap 4, and tolerance 0.2. Start with `--group-level`.

Sources in this checkout: [steering semantics](STEERING.md),
[CLI options](trajectory_editor/episode_cli.py),
[group controller](trajectory_editor/group_control.py),
[manual group fitter](trajectory_editor/online_learning.py), and
[latent learner](trajectory_editor/latent_preference.py).
