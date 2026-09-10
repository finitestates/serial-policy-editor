"""llama-cpp-python adapter for the policy editor."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .domain import EditorError
from .episode_backend import CacheMode, EpisodeBackend, validate_cache_mode


# Public spelling retained because backend_factory and a few user scripts used it.
Decoder = EpisodeBackend

KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")


@dataclass(frozen=True)
class LlamaCppSettings:
    n_ctx: int = 2048
    n_batch: int = 256
    flash_attn: bool = True
    use_mmap: bool = True
    use_mlock: bool = False
    n_ubatch: int | None = None
    n_threads: int | None = None
    n_threads_batch: int | None = None
    n_gpu_layers: int | None = None
    split_mode: int | None = None
    main_gpu: int | None = None
    tensor_split: tuple[float, ...] | None = None
    offload_kqv: bool | None = None
    type_k: str | int | None = None
    type_v: str | int | None = None
    numa: bool | int | None = False
    rope_scaling_type: int | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None

    def __post_init__(self) -> None:
        if type(self.n_ctx) is not int or self.n_ctx < 1:
            raise EditorError("n_ctx must be a positive integer")
        if type(self.n_batch) is not int or self.n_batch < 1:
            raise EditorError("n_batch must be a positive integer")
        for name in ("flash_attn", "use_mmap", "use_mlock"):
            if type(getattr(self, name)) is not bool:
                raise EditorError(f"{name} must be a boolean")
        for name in ("n_ubatch", "n_threads", "n_threads_batch"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise EditorError(f"{name} must be a positive integer or auto")
        if self.n_gpu_layers is not None and type(self.n_gpu_layers) is not int:
            raise EditorError("n_gpu_layers must be an integer or auto")
        if self.main_gpu is not None and type(self.main_gpu) is not int:
            raise EditorError("main_gpu must be an integer or auto")
        for name in ("type_k", "type_v"):
            value = getattr(self, name)
            if value is not None and type(value) is not int and value not in KV_CACHE_TYPES:
                raise EditorError(f"{name} must be one of {KV_CACHE_TYPES} or a GGML type integer")
        if not self.flash_attn and self.type_v in ("q8_0", "q4_0"):
            raise EditorError("Quantized V cache requires Flash Attention; remove --no-flash-attn or use --cache-type-v f16")
        if self.tensor_split is not None and (
            not self.tensor_split
            or any(
                type(value) not in (int, float) or value < 0
                for value in self.tensor_split
            )
        ):
            raise EditorError(
                "tensor_split must be a nonempty list of nonnegative numbers"
            )


class LlamaCppDecoder:
    """Thin adapter over llama-cpp-python's token evaluation API."""

    def __init__(
        self,
        model_path: Path,
        settings: LlamaCppSettings,
        *,
        cache_mode: CacheMode = "auto",
    ) -> None:
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required; install this package with [llama]"
            ) from exc
        self._llama_cpp = llama_cpp
        self.model_path = Path(model_path)
        self.settings = settings
        self._cache_mode = validate_cache_mode(cache_mode)
        self._cache_enabled = self._cache_mode == "auto"
        if not self.model_path.is_file():
            raise RuntimeError(f"model file does not exist: {self.model_path}")
        # SPE samples the returned logits itself. A llama.cpp RNG seed would
        # describe an unused sampler and could contradict the active SPE seed.
        options = {
            "model_path": str(self.model_path),
            "n_ctx": settings.n_ctx,
            "n_batch": settings.n_batch,
            "flash_attn": settings.flash_attn,
            "use_mmap": settings.use_mmap,
            "use_mlock": settings.use_mlock,
            "logits_all": False,
            "verbose": False,
            "n_ubatch": settings.n_ubatch,
            "n_threads": settings.n_threads,
            "n_threads_batch": settings.n_threads_batch,
            "n_gpu_layers": settings.n_gpu_layers,
            "split_mode": settings.split_mode,
            "main_gpu": settings.main_gpu,
            "tensor_split": list(settings.tensor_split) if settings.tensor_split else None,
            "offload_kqv": settings.offload_kqv,
            "type_k": settings.type_k,
            "type_v": settings.type_v,
            "numa": settings.numa,
            "rope_scaling_type": settings.rope_scaling_type,
            "rope_freq_base": settings.rope_freq_base,
            "rope_freq_scale": settings.rope_freq_scale,
        }
        for name in ("type_k", "type_v"):
            value = options[name]
            if isinstance(value, str):
                constant = "GGML_TYPE_" + value.upper()
                if not hasattr(llama_cpp, constant):
                    raise EditorError(f"Installed llama-cpp-python does not support cache type {value}")
                options[name] = getattr(llama_cpp, constant)
        if not settings.flash_attn and options["type_v"] is not None:
            quantized_types = {
                getattr(llama_cpp, "GGML_TYPE_" + name.upper(), None)
                for name in KV_CACHE_TYPES if name != "f16"
            }
            if options["type_v"] in quantized_types:
                raise EditorError("Quantized V cache requires Flash Attention")
        self._model = Llama(**{k: v for k, v in options.items() if v is not None})
        n_vocab = getattr(self._model, "n_vocab", None)
        self._vocabulary_size = int(n_vocab() if callable(n_vocab) else n_vocab)
        if self._vocabulary_size < 1:
            raise RuntimeError("llama.cpp reported no addressable vocabulary")
        self._fallback_eog_ids: set[int] = set()
        for method_name in ("token_eos", "token_eot", "token_eom"):
            method = getattr(self._model, method_name, None)
            if callable(method):
                try:
                    value = int(method())
                except (TypeError, ValueError):
                    continue
                if value >= 0:
                    self._fallback_eog_ids.add(value)
        self._tokens: list[int] = []

    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def reset(self, prefix_token_ids: list[int]) -> None:
        if not prefix_token_ids:
            raise RuntimeError("decoder prefix cannot be empty")
        values = [int(value) for value in prefix_token_ids]
        self._model.reset()
        self._model.eval(values)
        self._tokens = values

    def eval(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        values = [int(value) for value in token_ids]
        self._tokens.extend(values)
        if self._cache_enabled:
            self._model.eval(values)
        else:
            self._model.reset()
            self._model.eval(self._tokens)

    def branch_to_prefix(self, prefix_token_ids: list[int]) -> None:
        """Move to an existing prefix, reusing the current cache when possible."""
        values = [int(value) for value in prefix_token_ids]
        if not values:
            raise RuntimeError("decoder prefix cannot be empty")
        if not self._cache_enabled:
            self.reset(values)
            return
        if len(values) > len(self._tokens) or self._tokens[: len(values)] != values:
            self.reset(values)
            return
        if len(values) == len(self._tokens):
            return
        try:
            # Keep the cache through the token before the final retained token,
            # then evaluate that one token again to refresh final-position logits.
            retained_before_last = len(values) - 1
            removed = self._model._ctx.kv_cache_seq_rm(
                -1, retained_before_last, -1
            )
            if not removed:
                self.reset(values)
                return
            self._model.n_tokens = retained_before_last
            self._model._requires_eval = True
            self._model.eval([values[-1]])
            self._tokens = values
        except (AttributeError, RuntimeError, TypeError):
            # Cache positioning is an optimization. If the installed binding
            # cannot safely truncate its sequence, preserve semantics via reset.
            self.reset(values)

    def last_logits(self) -> np.ndarray:
        model = self._model
        context = model._ctx.ctx if hasattr(model, "_ctx") else model.ctx
        get_ith = getattr(self._llama_cpp, "llama_get_logits_ith", None)
        pointer = (
            get_ith(context, -1)
            if callable(get_ith)
            else self._llama_cpp.llama_get_logits(context)
        )
        if not pointer:
            raise RuntimeError("llama.cpp returned no final-position logits")
        return np.ctypeslib.as_array(pointer, shape=(self._vocabulary_size,)).astype(
            np.float32, copy=True
        )

    def tokenize(
        self, text: str, *, add_bos: bool = False, special: bool = False
    ) -> list[int]:
        return [
            int(value)
            for value in self._model.tokenize(
                text.encode("utf-8"), add_bos=add_bos, special=special
            )
        ]

    def render(self, token_ids: list[int], *, special: bool = False) -> str:
        if not token_ids:
            return ""
        return self._model.detokenize(
            [int(value) for value in token_ids], special=special
        ).decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str:
        text = self.render([int(token_id)], special=self.is_eog(token_id))
        return text or ("<EOG>" if self.is_eog(token_id) else "")

    def is_eog(self, token_id: int) -> bool:
        try:
            return bool(
                self._llama_cpp.llama_vocab_is_eog(
                    self._model._model.vocab, int(token_id)
                )
            )
        except (AttributeError, TypeError):
            return int(token_id) in self._fallback_eog_ids

    def eog_token_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._fallback_eog_ids))

    def provenance(self, *, include_model_sha256: bool = True) -> dict[str, Any]:
        # Provenance identifies the model/runtime needed to interpret the run;
        # private evaluation state is neither evidence nor episode identity.
        stat = self.model_path.stat()
        requested = asdict(self.settings)
        if requested.get("tensor_split") is not None:
            requested["tensor_split"] = list(requested["tensor_split"])
        return {
            "backend": "llama.cpp",
            "adapter": "llama-cpp-python",
            "model_path": str(self.model_path.resolve()),
            "filename": self.model_path.name,
            "file_size_bytes": int(stat.st_size),
            "vocabulary_size": self.vocabulary_size(),
            "llama_cpp_python_version": getattr(self._llama_cpp, "__version__", None),
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "runtime_configuration": requested,
        }
