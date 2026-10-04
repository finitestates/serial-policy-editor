# Argmax controller profiles

These YAML files are current controller profiles, not teacher tapes. Pass one
with `--profile` alongside your model and prompt, for example:

```sh
policy-editor --model /path/model.gguf --new-prompt 'A story' --profile examples/argmax.yaml
```

`argmax.yaml` uses full-vocabulary plain argmax. `gumbel-control.yaml` adds unit
Gumbel noise to every eligible token. `eligible-selective.yaml` restricts
eligibility to 20 candidates and perturbs only the leading 5. The full-vocabulary
`gaussian-robustness.yaml` perturbs only the leading 5, so unchanged competitors
can win. Examples have not been executed during this documentation cut.
