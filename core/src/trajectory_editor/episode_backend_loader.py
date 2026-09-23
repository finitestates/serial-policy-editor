"""Load a model backend from explicit or persisted episode launch settings."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .backend_factory import BACKEND_NAMES, create_backend
from .core.errors import EditorError
from .decoder import LlamaCppSettings
from .transformers_backend import TransformersSettings
from .terminal_contracts import PromptRequest


class BackendLoadIO(Protocol):
    """The small interactive surface used when saved model loading needs help."""

    def prompt(self, request: PromptRequest) -> str | None: ...

    def write(self, message: str) -> None: ...


def load_backend(args: argparse.Namespace) -> Any:
    """Turn launch arguments into one loaded backend instance."""

    args.backend = args.backend or "llama.cpp"
    if args.model is None:
        raise EditorError("--model is required to start, resume, fork, or replay")
    llama = LlamaCppSettings(
        type_k=args.type_k,
        type_v=args.type_v,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads_batch,
        n_gpu_layers=args.n_gpu_layers,
        main_gpu=args.main_gpu,
        flash_attn=not args.no_flash_attn,
        use_mmap=not args.no_mmap,
        use_mlock=args.use_mlock,
    )
    transformers = TransformersSettings(
        device=args.transformers_device,
        dtype=args.transformers_dtype,
        device_map=args.transformers_device_map,
        attention_implementation=args.transformers_attention_implementation,
        quantization_method=args.transformers_quantization,
        trust_remote_code=args.transformers_trust_remote_code,
        use_fast_tokenizer=not args.transformers_slow_tokenizer,
        torch_num_threads=args.transformers_torch_threads,
        torch_num_interop_threads=args.transformers_torch_interop_threads,
    )
    print(f"Loading model with {args.backend}: {args.model} ...", flush=True)
    result = create_backend(
        args.backend,
        args.model,
        llama_settings=llama,
        transformers_settings=transformers,
        cache_mode=args.cache,
    )
    print("Model loaded.", flush=True)
    return result


def load_episode_backend(
    args: argparse.Namespace,
    source: Mapping[str, Any] | None,
    io: BackendLoadIO,
    *,
    use_saved: bool = False,
    current_backend: Any | None = None,
    current_provenance: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], bool]:
    """Load saved execution context and confirm intentional model changes."""

    selected = copy.copy(args)
    saved = source["backend"] if source else {}
    old_path = saved.get("model_path")
    if use_saved:
        selected.model = None
        selected.backend = None
    if selected.model is None and old_path:
        selected.model = Path(old_path)
    selected.backend = selected.backend or saved.get("backend") or "llama.cpp"
    if selected.backend not in BACKEND_NAMES:
        selected.backend = args.backend or "llama.cpp"
    # Persisted launch options avoid reconstructing device/quantization settings
    # from diagnostic effective values. Explicit launch flags take precedence.
    explicit = getattr(args, "_explicit_options", set()) if not use_saved else set()
    saved_options = dict(saved.get("load_options", {}))
    if not saved_options:
        for key, value in saved.get("runtime_configuration", {}).items():
            if saved.get("backend") == "transformers":
                key = "transformers_" + key
            elif key in {"flash_attn", "use_mmap"}:
                key, value = {
                    "flash_attn": "no_flash_attn",
                    "use_mmap": "no_mmap",
                }[key], not value
            if hasattr(selected, key) and key != "seed":
                saved_options[key] = value
    for key, value in saved_options.items():
        if key not in explicit:
            setattr(selected, key, value)
    while True:
        if selected.model is None:
            path = io.prompt(PromptRequest("Saved model location unavailable. Model path (Enter cancels)> "))
            if not path:
                raise EditorError("model loading cancelled")
            selected.model = Path(path).expanduser()
        changed = bool(source and old_path and (
            Path(old_path).resolve() != selected.model.resolve()
            or (
                saved.get("backend") in BACKEND_NAMES
                and saved.get("backend") != selected.backend
            )
        ))
        if changed:
            answer = io.prompt(PromptRequest(
                f"Previously used {old_path} ({saved.get('backend')}). Continue with "
                f"{selected.model} ({selected.backend}) in a new linked episode? [y/N]> "
            ))
            if not answer or answer.strip().lower() not in {"y", "yes"}:
                raise EditorError("model change cancelled")
        try:
            if (
                current_backend is not None
                and current_provenance
                and current_provenance.get("model_path")
                == str(selected.model.resolve())
                and current_provenance.get("backend") == selected.backend
                and all(
                    getattr(selected, key, None) == value
                    for key, value in current_provenance.get("load_options", {}).items()
                )
            ):
                return current_backend, dict(current_provenance), changed
            backend = load_backend(selected)
            provenance = dict(backend.provenance(include_model_sha256=True))
            provenance["model_path"] = str(selected.model.resolve())
            provenance["load_options"] = {
                key: value
                for key, value in vars(selected).items()
                if key.startswith("transformers_")
                or key in {
                    "n_ctx",
                    "n_batch",
                    "n_ubatch",
                    "n_threads",
                    "n_threads_batch",
                    "n_gpu_layers",
                    "main_gpu",
                    "no_flash_attn",
                    "no_mmap",
                    "use_mlock",
                    "cache",
                    "type_k",
                    "type_v",
                }
            }
            return backend, provenance, changed
        except (EditorError, OSError, RuntimeError) as exc:
            if not source:
                raise
            io.write(f"Could not load {selected.model}: {exc}")
            path = io.prompt(PromptRequest("Replacement model path (Enter cancels)> "))
            if not path:
                raise EditorError("model loading cancelled") from exc
            kind = io.prompt(PromptRequest("Backend: llama.cpp or transformers (Enter keeps current)> "))
            if kind:
                if kind.strip() not in BACKEND_NAMES:
                    io.write("Unknown backend.")
                    continue
                selected.backend = kind.strip()
            selected.model = Path(path).expanduser()


def cfg_required(sampling, *, plan=None, historical_sampling=()) -> bool:
    """Runtime setup must provision guidance for reachable sampler controls."""
    configurations = [sampling, *historical_sampling]
    if plan is not None and plan.follow_source_sampling:
        configurations.extend(plan.context.sampling)
        configurations.append(plan.final_sampling)
    return any(
        config is not None and config.cfg_unconditional_prompt is not None
        for config in configurations
    )


def load_cfg_guidance_backend(
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
) -> Any:
    """Load the active model again for CFG's unconditional branch."""

    selected = copy.copy(args)
    selected.model = Path(provenance["model_path"])
    selected.backend = provenance["backend"]
    for key, value in provenance.get("load_options", {}).items():
        if hasattr(selected, key):
            setattr(selected, key, value)
    return load_backend(selected)


__all__ = [
    "BackendLoadIO",
    "load_backend",
    "load_cfg_guidance_backend",
    "load_episode_backend",
]
