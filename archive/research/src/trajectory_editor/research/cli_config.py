"""Command-line options owned by the optional research surface.

The core episode command deliberately knows nothing about these options.  The
research entry point adds them to the core parser before handing the parsed
namespace to the shared episode coordinator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DECAY_ON = ("update", "rejection", "evidence")
WRITE_REDUCTIONS = ("sum", "mean", "sqrt")
REJECTION_TARGETS = ("proposal", "sampler")
DEFAULT_PROJECTION_CHUNK_SIZE = 8192
TOKEN_PREFERENCE_FEATURE_SCHEMES = (
    "random-projection-unit-v1",
    "whitened-projection-v2",
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _add_learning_experiment_flags(parser: argparse._ArgumentGroup, prefix: str) -> None:
    parser.add_argument(
        f"--{prefix}-decay-on",
        choices=DECAY_ON,
        default="update",
        help="forget on every update (default), proposal rejection, or gate-admitted evidence; writes decay at most once",
    )
    parser.add_argument(
        f"--{prefix}-write-reduction",
        choices=WRITE_REDUCTIONS,
        default="sum",
        help="scale a write's evidence by 1, 1/N, or 1/sqrt(N); N counts gate-admitted tokens",
    )
    parser.add_argument(
        f"--{prefix}-rejection-target",
        choices=REJECTION_TARGETS,
        default="proposal",
        help="contrast teacher corrections with the sampled proposal or the frozen sampler distribution; requires nonzero rejection strength",
    )


def add_research_arguments(parser: argparse.ArgumentParser) -> None:
    """Add research-only controls to an already-built core parser."""

    parser.add_argument(
        "--biases",
        type=Path,
        help="load a compiled bias preset",
    )
    parser.add_argument(
        "--bias-catalog",
        type=Path,
        help="load a model-matched human-readable bias catalog for b name and b @name",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        help="load semantic term/group YAML directly",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="load standalone relative lexical weights from YAML",
    )
    parser.add_argument(
        "--reference-strength",
        type=float,
        help="overall lexical influence (default: 0.25)",
    )
    parser.add_argument(
        "--group-level",
        type=float,
        default=1.0,
        help="appearance objective level; 1 requests twice/half the baseline odds",
    )
    parser.add_argument(
        "--group-control-scheme",
        choices=("appearance-feedback-v1", "appearance-rate-v2"),
        default=None,
        help="appearance controller mathematics",
    )
    parser.add_argument(
        "--reference-prior",
        choices=(
            "off",
            "active", "active-exit",
            "ballistic-active", "ballistic-active-exit",
            "global", "global-exit",
            "ballistic-global", "ballistic-global-exit",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--reference-prior-strength", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--reference-prior-attraction", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--reference-prior-exit-strength", type=float, help=argparse.SUPPRESS)

    learning = parser.add_argument_group("online learning")
    learning.add_argument(
        "--online-learning",
        "--online-learning-enabled",
        dest="online_learning",
        action="store_true",
        help="fit explicitly learnable manual group weights to teacher selections; appearance objectives run independently",
    )
    learning.add_argument("--learning-rate", type=float, default=0.05)
    learning.add_argument("--epsilon", "--learning-epsilon", dest="learning_epsilon", type=float, default=0.05)
    learning.add_argument("--max-step", "--learning-max-step", dest="learning_max_step", type=float, default=0.25)
    learning.add_argument("--min-bias", "--learning-min-bias", dest="learning_min_bias", type=float, default=-4.0)
    learning.add_argument("--max-bias", "--learning-max-bias", dest="learning_max_bias", type=float, default=4.0)
    learning.add_argument("--learning-severity-cap", type=_positive_int, default=1000)
    learning.add_argument("--learning-dead-zone-rank", type=_nonnegative_int, default=0)
    learning_severity = learning.add_mutually_exclusive_group()
    learning_severity.add_argument(
        "--learning-no-severity-attenuation",
        dest="learning_no_severity_attenuation",
        action="store_true",
        help="use full severity for every teacher selection (the default)",
    )
    learning_severity.add_argument(
        "--learning-severity-attenuation",
        dest="learning_no_severity_attenuation",
        action="store_false",
        help="restore legacy rank dead-zone and severity attenuation",
    )
    learning.set_defaults(learning_no_severity_attenuation=True)
    learning.add_argument("--learning-rejection-strength", type=float, default=0.0)
    learning.add_argument("--learning-decay", type=float, default=0.0)
    _add_learning_experiment_flags(learning, "learning")
    learning.add_argument(
        "--learning-gate",
        choices=("rank", "sampler"),
        default="rank",
        help="sampler replaces rank severity: learn at full severity only from filtered-out tokens; decay is unchanged",
    )
    learning.add_argument(
        "--learnable-groups",
        nargs="+",
        metavar="GROUP",
        help="restrict online learning to these groups; opt them in with b GROUP learn on",
    )
    write_learning = learning.add_mutually_exclusive_group()
    write_learning.add_argument(
        "--learn-from-write",
        dest="learn_from_write",
        action="store_true",
        help="learn from each token in live typed writes (the default)",
    )
    write_learning.add_argument(
        "--no-learn-from-write",
        dest="learn_from_write",
        action="store_false",
        help="restore the legacy behavior that does not learn from typed writes",
    )
    learning.set_defaults(learn_from_write=True)

    preference = parser.add_argument_group("token preference learning")
    preference.add_argument(
        "--token-preference",
        "--token-preference-enabled",
        dest="token_preference",
        action="store_true",
        help="learn an anonymous token preference vector from live raw-rank selections (off by default)",
    )
    preference.add_argument("--token-preference-dimension", type=int, default=64)
    preference.add_argument("--token-preference-learning-rate", type=float, default=0.05)
    preference.add_argument("--token-preference-strength", type=float, default=1.0)
    preference.add_argument(
        "--token-preference-feature-scheme",
        choices=TOKEN_PREFERENCE_FEATURE_SCHEMES,
        default=None,
        help="preference feature coordinate scheme; v1 is the replay-compatible default",
    )
    preference.add_argument("--token-preference-whitening-ridge", type=float, default=None)
    preference.add_argument(
        "--token-preference-learning-scheme",
        choices=("sgd-v1", "fisher-kl-v2"),
        default=None,
        help="token preference memory update geometry",
    )
    preference.add_argument(
        "--token-preference-influence-mode",
        choices=("manual", "kl"),
        default=None,
        help="manual actuator strength or automatic KL-calibrated gain",
    )
    preference.add_argument("--token-preference-influence-kl", type=float, default=None)
    preference.add_argument("--token-preference-min-gain", type=float, default=None)
    preference.add_argument("--token-preference-max-gain", type=float, default=None)
    preference.add_argument("--token-preference-max-step", type=float, default=0.25)
    preference.add_argument("--token-preference-max-norm", type=float, default=4.0)
    preference.add_argument("--token-preference-decay", type=float, default=0.0)
    _add_learning_experiment_flags(preference, "token-preference")
    preference.add_argument("--token-preference-severity-cap", type=_positive_int, default=1000)
    preference_severity = preference.add_mutually_exclusive_group()
    preference_severity.add_argument(
        "--token-preference-no-severity-attenuation",
        dest="token_preference_no_severity_attenuation",
        action="store_true",
        help="use full severity for every teacher selection (the default)",
    )
    preference_severity.add_argument(
        "--token-preference-severity-attenuation",
        dest="token_preference_no_severity_attenuation",
        action="store_false",
        help="restore legacy rank dead-zone and severity attenuation",
    )
    preference.set_defaults(token_preference_no_severity_attenuation=True)
    preference.add_argument("--token-preference-dead-zone-rank", type=_nonnegative_int, default=0)
    preference.add_argument(
        "--token-preference-learning-gate",
        choices=("rank", "sampler"),
        default="rank",
        help="sampler replaces rank severity: learn at full severity only from filtered-out tokens; decay is unchanged",
    )
    preference.add_argument("--token-preference-rejection-strength", type=float, default=0.0)
    preference.add_argument("--token-preference-fast-slow", action="store_true")
    preference.add_argument("--token-preference-fast-learning-rate", type=float)
    preference.add_argument("--token-preference-fast-decay", type=float, default=0.10)
    preference.add_argument("--token-preference-fast-strength", type=float)
    preference.add_argument("--token-preference-fast-max-step", type=float)
    preference.add_argument("--token-preference-fast-max-norm", type=float)
    preference.add_argument("--token-preference-learning-metric", choices=("euclidean", "fisher"), default=None)
    preference.add_argument("--token-preference-learning-kl", type=float, default=None)
    preference.add_argument(
        "--token-preference-fast-learning-kl",
        type=float,
        default=None,
        help="fast-memory canonical learning KL budget (defaults to the slow budget)",
    )
    preference.add_argument("--token-preference-fisher-ridge", type=float, default=None)
    preference.add_argument("--token-preference-fisher-mode", choices=("diagonal", "full"), default=None)
    preference.add_argument("--token-preference-fisher-mass", type=float, default=None)
    preference.add_argument("--token-preference-fisher-max-support", type=_positive_int, default=None)
    token_preference_seeds = preference.add_mutually_exclusive_group()
    token_preference_seeds.add_argument("--token-preference-projection-seed", type=int)
    token_preference_seeds.add_argument(
        "--token-preference-random-projection-seed",
        dest="token_preference_random_projection_seed",
        action="store_true",
    )
    preference.add_argument(
        "--token-preference-projection-chunk-size",
        type=_positive_int,
        default=DEFAULT_PROJECTION_CHUNK_SIZE,
        help="rows projected at once when building token preference features; lower this to reduce peak memory at the cost of slower initialization",
    )

    projection = parser.add_argument_group("research projection")
    projection.add_argument(
        "--biases-only",
        action="store_true",
        help="with --project, emit a loadable JSON bias preset",
    )
    projection.add_argument(
        "--rules-only",
        action="store_true",
        help="with --biases-only, flatten named groups into ordinary logical rules",
    )
    projection.add_argument(
        "--editor-friendly",
        action="store_true",
        help="with --biases-only, emit standalone YAML group definitions for policy-editor-bias",
    )
    projection.add_argument("--annotations", choices=("none", "inline", "footnotes"), default="none")
    projection.add_argument("--with-loss", action="store_true")
    projection.add_argument("--with-rank", action="store_true")
    projection.add_argument("--with-policy-rank", action="store_true")
    projection.add_argument(
        "--full-evidence",
        action="store_true",
        help="footnote teacher-selected tokens with proposal agreement, NLL, and ranks",
    )
    projection.add_argument(
        "--with-model-probs",
        "--with-model",
        dest="with_model_probs",
        action="store_true",
        help="add raw-model and decoder probabilities to teacher-token footnotes",
    )
    projection.add_argument(
        "--with-lineage",
        action="store_true",
        help="append fork-family and replay metadata to a projection",
    )


__all__ = ["add_research_arguments"]
