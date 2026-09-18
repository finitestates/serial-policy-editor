# Policy Editor Research

This optional subproject contains learning, reference priors, bias catalogs,
token preference, instrumentation surfaces, post-output vector construction,
and other experimental vector work. It adapts historical records to the core
replay contract instead of being imported by the core runner.

Research notes and experiments live under [`docs/`](docs/). This build root is
independently installable as `policy-editor-research`; it depends on the core
distribution and supplies the adapter that reconstructs historical research
sampler records.
