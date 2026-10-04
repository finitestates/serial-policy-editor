# Reference kernel status on the argmax branch

The independent kernel and its tests still implement the previous engine:
categorical CDF draws, retired probability filters and old production APIs.
They have not been ported or executed for `policy-editor-argmax`, and are not an
oracle for its contracts. Shared perturbation formulas alone do not establish
pipeline or replay parity.

[Historical usage](LEGACY_USAGE.md) preserves the old instructions and scope.
Its production-importing wrappers may no longer run on this branch.

The [testing harness plan](../docs/ARGMAX_TEST_HARNESS_PLAN.md) calls for an
independent score/membership/noise/winner implementation before parity claims
resume. Production helpers must not be used to calculate its expected winners.
Model parity also needs matching tokenizer IDs, model settings and evaluation
boundaries; synthetic parity alone does not certify backend behavior.
