# Rust port experiments

Each targeted rewrite gets its own directory under this folder. The current
experiment is [`sampler/`](sampler/README.md), a Rust crate with its own
`Cargo.toml`, lockfile, Python binding, fixtures, and instructions.

For another rewrite, create a sibling directory such as
`experiments/rust-port/<component>/` and give it its own `Cargo.toml` and
README. That keeps each prototype's dependencies, build commands, tests, and
Python boundary easy to understand and lets experiments be built and reviewed
independently.

Rust calls each package a **crate**. A crate can be a reusable library or an
executable. A **Cargo workspace** is an optional way to coordinate several
crates, share dependency versions, and run commands across them. These
experiments do not need to share a workspace just because they live under the
same directory. Keep them independent at first; make a workspace later if
shared code or coordinated commands provide a concrete benefit. If a workspace
is added, document its root commands and lockfile location here.

The experiments are not part of the released Python packages. Promote a port
into production only after its behavior, build/install path, and maintenance
cost have been reviewed.
