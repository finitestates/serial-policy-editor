"""Storage-neutral construction of an unrelated live episode root."""

from __future__ import annotations

from dataclasses import replace

from .core.errors import EditorError
from .episode_engine import EpisodeEngine


def fresh_root_from(engine: EpisodeEngine, prompt: str) -> EpisodeEngine:
    """Create a new prompt root on the engine's already-loaded backends.

    The source engine contributes only the loaded primary/guidance backends,
    sampler configuration, and configured tranche allowance.  The new engine
    tokenizes ``prompt`` itself, which gives it a new stream fingerprint and a
    zero coordinate without carrying any source episode state across.
    """

    if not isinstance(engine, EpisodeEngine):
        raise TypeError("engine must be an EpisodeEngine")
    if not isinstance(prompt, str) or not prompt:
        raise EditorError("prompt must be a nonempty string")
    return EpisodeEngine(
        engine.backend,
        sampling=replace(engine.sampling),
        max_tokens=engine.max_tokens,
        initial_text=prompt,
        coordinate_offset=0,
        guidance_backend=engine.guidance_backend,
        guidance_initial_token_ids=(
            engine.guidance_initial_token_ids or None
        ),
    )


__all__ = ["fresh_root_from"]
