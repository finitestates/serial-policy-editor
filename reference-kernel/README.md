# Independent argmax reference kernel

`reference_kernel.py` independently implements policy adjustments, temperature,
top-k/logit-gap eligibility, plain argmax and replay-addressed noise. Selective
noise chooses candidates by pre-noise score with token-ID ties; unchanged
eligible candidates still compete. Default selection is full-vocabulary argmax.
There is no categorical token CDF or retired probability-filter pipeline.

The kernel imports standard-library modules only. Its expected scores, membership
and winners do not call production sampling helpers. Optional `probabilities`
on its Distribution record normalize eligible pre-noise scores; these are
eligible-softmax diagnostics, not general winner probabilities.

From the repository root:

```sh
PYTHONPATH=core/src python -m pytest -q reference-kernel
python reference-kernel/draw_coordinates_demo.py
python reference-kernel/reference_kernel_oracle.py atomic \
  --backend llama --model /path/model.gguf --prefix 'A story' \
  --draw-kernel gumbel-max --top-k none --selective-noise-k 5 --detail tokens
```

The coordinate demonstration holds scores fixed and changes the boundary under
Gumbel noise. `--detail tokens` avoids probability diagnostics. `probabilities`
and `full` request eligible-softmax explicitly. Display limits do not change
eligibility, noise application or selection.

`reference_kernel_oracle.py` supports atomic and pattern full-prefix model
observations. Match token IDs, root fingerprint, boundary, model settings,
precision and tokenizer representation when comparing with production. Synthetic
parity does not certify prefill/incremental model parity. CFG, grouped phrase
biases, vectors and beam remain outside this minimal model CLI's scope.

The [validation record](../docs/ARGMAX_HARNESS_UPDATE.md) reports local parity
and skipped real-model checks. [Historical usage](LEGACY_USAGE.md) preserves the
previous engine's instructions; its removed flags and CDF expectations are
historical, not current guidance.
