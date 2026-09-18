# Policy Editor Vectors

This optional subproject is reserved for conventional hidden-state
activation/steering vectors: loading externally produced artifacts, importing
llama.cpp cvector files, creating vectors from prompt pairs, and inspecting
those artifacts. Experimental vector formats and analysis tools do not belong
in this package.

Install it after `policy-editor-core` when you want `policy-editor-vector`.
The core runtime can load a compatible artifact without this package; this
package adds production and inspection commands only.

```bash
python -m pip install './core[llama]' ./vector
```

For Transformers-backed vector production, install either
`./core[transformers]` or the explicit `./core[transformers-accelerate]` extra
before installing this package.

Post-output vectors, including pre-normalization output-head directions, are
not part of the conventional vector package.
