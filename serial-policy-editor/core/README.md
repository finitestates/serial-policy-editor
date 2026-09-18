# Policy Editor Core

This subproject contains the standalone runtime: the menu-driven episode
editor, replay/resume/fork/rewind behavior, persistence, supported model
backends, core sampler, bias rules, and episode projector.

Install it directly from this repository with:

```bash
python -m pip install ./core
```

The base install does not install an inference backend. Choose one explicitly:

```bash
python -m pip install './core[llama]'
python -m pip install './core[transformers]'
python -m pip install './core[transformers-accelerate]'
```

The ordinary Transformers extra is sufficient for basic CPU/GPU inference and
does not install Accelerate. The `transformers-gguf` and `transformers-bnb`
extras are explicit heavier paths.

Core can load compatible externally produced steering-vector artifacts, but
vector creation and inspection are provided by the separate `vector/` package.
