"""llama-cpp-python adapter for the policy editor."""

from __future__ import annotations

from contextlib import nullcontext

import codecs
import ctypes
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core.errors import EditorError
from .core.backend import CacheMode, InferenceBackend, validate_cache_mode
from .core.backend_position import BackendPosition, compare_backend_position
from .model_hash import sha256_path


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


def _llama_options(
    llama_cpp: Any,
    model_path: Path,
    settings: LlamaCppSettings,
) -> dict[str, Any]:
    """Build options for the inference context."""
    options = {
        "model_path": str(model_path),
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
                raise EditorError(
                    f"Installed llama-cpp-python does not support cache type {value}"
                )
            options[name] = getattr(llama_cpp, constant)
    if not settings.flash_attn and options["type_v"] is not None:
        quantized_types = {
            getattr(llama_cpp, "GGML_TYPE_" + name.upper(), None)
            for name in KV_CACHE_TYPES
            if name != "f16"
        }
        if options["type_v"] in quantized_types:
            raise EditorError("Quantized V cache requires Flash Attention")
    return {key: value for key, value in options.items() if value is not None}


def _llama_tokenizer_id(
    model: Any, vocabulary_size: int, eog_token_ids: tuple[int, ...]
) -> str:
    digest = hashlib.sha256()
    digest.update(b"serial-policy-editor-llama-tokenizer-v1\0")
    metadata = getattr(model, "metadata", None)
    if callable(metadata):
        try:
            metadata = metadata()
        except (TypeError, RuntimeError):
            metadata = None
    tokenizer_metadata = {}
    if isinstance(metadata, dict):
        tokenizer_metadata = {
            str(key): value
            for key, value in metadata.items()
            if str(key).startswith("tokenizer.")
        }
    pieces = tokenizer_metadata.get("tokenizer.ggml.tokens")
    if isinstance(pieces, (list, tuple)) and len(pieces) == vocabulary_size:
        encoded_pieces = (str(piece).encode("utf-8") for piece in pieces)
    else:
        encoded_pieces = (
            bytes(model.detokenize([token_id], special=True))
            for token_id in range(vocabulary_size)
        )
    for token_id, piece in enumerate(encoded_pieces):
        digest.update(token_id.to_bytes(8, "little", signed=False))
        digest.update(len(piece).to_bytes(8, "little", signed=False))
        digest.update(piece)
    if tokenizer_metadata:
        digest.update(
            json.dumps(
                tokenizer_metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
    digest.update(json.dumps(eog_token_ids, separators=(",", ":")).encode())
    return digest.hexdigest()


class _LlamaCppTextStream:
    """Incremental UTF-8 decoding over newly detokenized llama.cpp pieces."""

    def __init__(self, model: Any, *, special: bool) -> None:
        self._model = model
        self._special = special
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def append(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        raw = self._model.detokenize(
            [int(value) for value in token_ids], special=self._special
        )
        return self._decoder.decode(raw, final=False)


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
        options = _llama_options(llama_cpp, self.model_path, settings)
        self._model = Llama(**options)
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
        self._restored_logits: np.ndarray | None = None
        self._last_logits_cache: np.ndarray | None = None
        self._speculation_prefix: tuple[int, ...] | None = None
        self._speculation_logits: np.ndarray | None = None
        self._token_embedding_matrix_cache: np.ndarray | None = None
        self._activation_logit_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self._model_sha256_cache: str | None = None
        self._tokenizer_id_cache = _llama_tokenizer_id(
            self._model, self._vocabulary_size, tuple(sorted(self._fallback_eog_ids))
        )
        # Set only by the optional real-model harness. Normal inference has no
        # measurement object or extra synchronization.
        self._real_model_probe = None

    def _measured_eval(self, values: list[int], kind: str, context_length: int) -> None:
        # A restored snapshot supplies its saved final logits until the context
        # is evaluated again. Once evaluation starts, the native output buffer
        # becomes authoritative again.
        self._restored_logits = None
        self._last_logits_cache = None
        probe = getattr(self, "_real_model_probe", None)
        interval = probe.model_call(kind, len(values), context_length) if probe is not None else nullcontext()
        with interval:
            self._model.eval(values)

    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def reset(self, prefix_token_ids: list[int]) -> None:
        if self._speculation_prefix is not None:
            self.rollback_speculation()
        if not prefix_token_ids:
            raise RuntimeError("decoder prefix cannot be empty")
        values = [int(value) for value in prefix_token_ids]
        self._restored_logits = None
        self._model.reset()
        self._measured_eval(values, "prefill", len(values))
        self._tokens = values

    def eval(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        if self._speculation_prefix is not None:
            self.rollback_speculation()
        self._eval_tokens(token_ids)

    def _eval_tokens(self, token_ids: list[int]) -> None:
        values = [int(value) for value in token_ids]
        self._tokens.extend(values)
        if self._cache_enabled:
            self._measured_eval(values, "incremental", len(self._tokens))
        else:
            self._model.reset()
            self._measured_eval(self._tokens, "rebuild", len(self._tokens))

    def speculate(self, token_id: int) -> bool:
        if not self._cache_enabled or not self._tokens:
            return False
        if self._speculation_prefix is not None:
            self.rollback_speculation()
        if not self._can_remove_cache():
            return False
        logits = getattr(self, "_last_logits_cache", None)
        if logits is None:
            logits = self.last_logits()
        self._speculation_prefix = tuple(self._tokens)
        self._speculation_logits = logits
        self._eval_tokens([int(token_id)])
        return True

    def commit_speculation(self) -> None:
        if self._speculation_prefix is None:
            return
        if len(self._tokens) != len(self._speculation_prefix) + 1:
            raise RuntimeError("speculative decoder position changed before commit")
        self._speculation_prefix = None
        self._speculation_logits = None

    def rollback_speculation(self) -> None:
        prefix = self._speculation_prefix
        if prefix is None:
            return
        if not self._remove_cache_from(len(prefix)):
            raise RuntimeError("llama.cpp rejected speculative cache rollback")
        self._model.n_tokens = len(prefix)
        self._model._requires_eval = True
        self._tokens = list(prefix)
        self._restored_logits = self._speculation_logits
        self._last_logits_cache = self._speculation_logits
        self._speculation_prefix = None
        self._speculation_logits = None

    def truncate_to(self, length: int) -> bool:
        if not self._cache_enabled:
            return False
        if type(length) is not int or length < 1 or length > len(self._tokens):
            raise RuntimeError("decoder cache truncation target is out of range")
        if self._speculation_prefix is not None:
            self.rollback_speculation()
        if length == len(self._tokens):
            return True
        if not self._remove_cache_from(length):
            return False
        self._model.n_tokens = length
        self._model._requires_eval = True
        self._tokens = self._tokens[:length]
        self._restored_logits = None
        self._last_logits_cache = None
        return True

    def position(self) -> BackendPosition:
        """Report the adapter prefix and llama.cpp's actual sequence range."""
        cursor = int(self._model.n_tokens)
        cache_start = cache_end = None
        get_memory = getattr(self._llama_cpp, "llama_get_memory", None)
        get_min = getattr(self._llama_cpp, "llama_memory_seq_pos_min", None)
        get_max = getattr(self._llama_cpp, "llama_memory_seq_pos_max", None)
        direct_position_api = all(
            callable(function) for function in (get_memory, get_min, get_max)
        )
        if direct_position_api:
            try:
                context = self._model._ctx.ctx if hasattr(self._model, "_ctx") else self._model.ctx
                memory = get_memory(context)
                minimum = int(get_min(memory, 0))
                maximum = int(get_max(memory, 0))
                if maximum >= 0 and minimum >= 0:
                    cursor = maximum + 1
                    cache_start, cache_end = minimum, maximum
                else:
                    cursor = 0
            except (AttributeError, RuntimeError, TypeError, ValueError):
                direct_position_api = False
        if not direct_position_api and cursor > 0:
            # Older llama-cpp-python bindings expose the maintained input count
            # but not the per-sequence memory-position functions.
            cache_start, cache_end = 0, cursor - 1
        logits_valid = bool(
            self._restored_logits is not None
            or self._last_logits_cache is not None
            or not getattr(self._model, "_requires_eval", False)
        )
        return BackendPosition(
            token_ids=tuple(self._tokens),
            cursor=cursor,
            cache_start=cache_start,
            cache_end=cache_end,
            cache_reusable=self._cache_enabled and cache_end is not None,
            logits_valid=logits_valid,
        )

    def _remove_cache_from(self, cursor: int) -> bool:
        context_wrapper = getattr(self._model, "_ctx", None)
        context = (
            context_wrapper.ctx
            if context_wrapper is not None and hasattr(context_wrapper, "ctx")
            else getattr(self._model, "ctx", None)
        )
        get_memory = getattr(self._llama_cpp, "llama_get_memory", None)
        remove_memory = getattr(self._llama_cpp, "llama_memory_seq_rm", None)
        if context is not None and callable(get_memory) and callable(remove_memory):
            memory = get_memory(context)
            if memory is not None:
                return bool(remove_memory(memory, -1, cursor, -1))
        remove_legacy = getattr(context_wrapper, "kv_cache_seq_rm", None)
        if callable(remove_legacy):
            return bool(remove_legacy(-1, cursor, -1))
        return False

    def _can_remove_cache(self) -> bool:
        context_wrapper = getattr(self._model, "_ctx", None)
        context = (
            context_wrapper.ctx
            if context_wrapper is not None and hasattr(context_wrapper, "ctx")
            else getattr(self._model, "ctx", None)
        )
        return (
            context is not None
            and callable(getattr(self._llama_cpp, "llama_get_memory", None))
            and callable(getattr(self._llama_cpp, "llama_memory_seq_rm", None))
        ) or callable(getattr(context_wrapper, "kv_cache_seq_rm", None))

    def _truncate_cache_to_cursor(self, cursor: int) -> None:
        if not self._remove_cache_from(cursor):
            raise RuntimeError("llama.cpp rejected cache truncation")
        self._model.n_tokens = cursor
        self._model._requires_eval = True
        self._tokens = self._tokens[:cursor]
        self._restored_logits = None
        self._last_logits_cache = None

    def branch_to_prefix(self, prefix_token_ids: list[int]) -> None:
        """Align to any related prefix by cropping at the longest shared prefix."""
        if self._speculation_prefix is not None:
            self.rollback_speculation()
        values = [int(value) for value in prefix_token_ids]
        if not values:
            raise RuntimeError("decoder prefix cannot be empty")
        position = self.position()
        comparison = compare_backend_position(position, values)
        if comparison.status == "aligned":
            return
        if (
            comparison.status == "unknown"
            or not position.cache_reusable
            or position.cache_start is None
        ):
            self.reset(values)
            return
        shared = comparison.common_prefix_length
        if shared <= position.cache_start or (
            shared == len(values)
            and position.cursor >= len(values)
            and shared - 1 <= position.cache_start
        ):
            # A windowed cache may no longer contain the history needed by a
            # shortened or divergent prefix.
            self.reset(values)
            return
        try:
            if shared == len(values) and position.cursor >= len(values):
                if position.cursor == len(values) and position.logits_valid:
                    self._model.n_tokens = position.cursor
                    self._tokens = values
                    return
                # A shortened prefix needs fresh final-position logits. Keep
                # its parent cache and evaluate the last retained token again.
                self._truncate_cache_to_cursor(shared - 1)
                self._measured_eval([values[-1]], "branch", len(values))
                self._tokens = values
                return
            retained = min(position.cursor, shared)
            if position.cursor > retained:
                self._truncate_cache_to_cursor(retained)
            else:
                if position.cursor != len(self._tokens):
                    self._model._requires_eval = True
                self._model.n_tokens = position.cursor
                self._tokens = values[:position.cursor]
            suffix = values[retained:]
            if suffix:
                self._measured_eval(suffix, "branch", len(values))
                self._tokens = values
        except (AttributeError, RuntimeError, TypeError, ValueError):
            probe = getattr(self, "_real_model_probe", None)
            if probe is not None:
                probe.cache_fallback("branch-cache-unavailable")
            self.reset(values)

    def last_logits(self) -> np.ndarray:
        if getattr(self, "_speculation_prefix", None) is not None:
            raise RuntimeError("cannot read logits while a speculative token is cached")
        restored_logits = getattr(self, "_restored_logits", None)
        if restored_logits is not None:
            self._last_logits_cache = restored_logits
            return restored_logits.copy()
        cached = getattr(self, "_last_logits_cache", None)
        if cached is not None:
            return cached.copy()
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
        logits = np.ctypeslib.as_array(pointer, shape=(self._vocabulary_size,)).astype(
            np.float32, copy=True
        )
        self._last_logits_cache = logits
        return logits.copy()

    def activation_width(self) -> int:
        """Return the final hidden/output-head width for this GGUF model."""
        return int(self._model.n_embd())

    def activation_control_vector_width(self) -> int:
        """Return the per-layer width expected by llama.cpp cvector data."""
        return self.activation_width()

    def activation_control_vector_layer_count(self) -> int:
        """Return the number of direction slots emitted by cvector-generator."""
        return self._control_vector_layer_count()

    def _model_layer_count(self) -> int:
        getter = getattr(self._llama_cpp, "llama_model_n_layer", None)
        if callable(getter):
            count = int(getter(self._model._model.model))
            if count < 1:
                raise RuntimeError("llama.cpp reported no decoder layers")
            return count
        raise RuntimeError("installed llama.cpp binding does not expose model layer count")

    def _control_vector_layer_count(self) -> int:
        """Return the number of canonical one-based decoder block outputs."""
        return self._model_layer_count()

    def set_activation_control_vector(
        self,
        vector,
        *,
        layer_start: int,
        layer_end: int,
        strength: float,
    ) -> None:
        """Install a canonical block-output vector on the live llama.cpp context.

        llama.cpp applies native slot ``s`` after zero-based decoder block
        ``s``.  Consequently canonical one-based block output ``N`` maps to
        native slot ``N - 1``.  Canonical layer 1 is unaddressable because the
        upstream native adapter reserves no slot 0; canonical layer N is
        fully steerable before final output normalization.
        """
        setter = getattr(self._llama_cpp, "llama_set_adapter_cvec", None)
        if not callable(setter):
            raise RuntimeError("installed llama.cpp binding does not expose control vectors")
        width = self.activation_control_vector_width()
        layer_count = self.activation_control_vector_layer_count()
        if layer_count < 2:
            raise RuntimeError(
                "llama.cpp control vectors expose no canonical block-output runtime layer"
            )
        runtime_start, runtime_end = 2, layer_count
        if (
            type(layer_start) is not int
            or type(layer_end) is not int
            or layer_start < runtime_start
            or layer_end < layer_start
            or layer_end > runtime_end
        ):
            raise RuntimeError(
                "control-vector canonical layer range is outside the loaded model "
                f"runtime range {runtime_start}..{runtime_end}"
            )
        values = np.asarray(vector, dtype=np.float32)
        if values.ndim != 1 or values.size != width * layer_count:
            raise RuntimeError("control-vector data does not match the loaded model")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("control-vector data is not finite")
        # The canonical vector has one chunk for every decoder block output.
        # Native llama.cpp cvector data begins at slot 1, so skip the
        # unaddressable canonical layer 1 chunk and pass layers 2..N
        # directly to native slots 1..N-1.  This preserves the final block's
        # pre-normalization injection point.
        native_values = np.ascontiguousarray(values[width:], dtype=np.float32)
        scaled = np.ascontiguousarray(native_values * float(strength), dtype=np.float32)
        context = self._model._ctx.ctx if hasattr(self._model, "_ctx") else self._model.ctx
        result = setter(
            context,
            scaled.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(scaled.size),
            width,
            layer_start - 1,
            layer_end - 1,
        )
        if int(result) != 0:
            raise RuntimeError(f"llama.cpp rejected the control vector (error {result})")

    def clear_activation_control_vector(self) -> None:
        """Remove any cvector from the live llama.cpp context."""
        setter = getattr(self._llama_cpp, "llama_set_adapter_cvec", None)
        if not callable(setter):
            return
        width = self.activation_control_vector_width()
        layer_count = self.activation_control_vector_layer_count()
        context = self._model._ctx.ctx if hasattr(self._model, "_ctx") else self._model.ctx
        result = setter(context, None, 0, width, 1, max(1, layer_count - 1))
        if int(result) != 0:
            raise RuntimeError(f"llama.cpp rejected clearing the control vector (error {result})")

    def _token_embedding_matrix(self) -> np.ndarray:
        if self._token_embedding_matrix_cache is not None:
            return self._token_embedding_matrix_cache
        binding = getattr(self._llama_cpp, "llama_cpp", self._llama_cpp)
        library = getattr(binding, "_lib", None)
        getter = getattr(
            library,
            "_Z24llama_model_get_tok_embdPK11llama_modelPf",
            None,
        )
        if getter is None:
            raise RuntimeError(
                "installed llama.cpp binding does not expose token embeddings"
            )
        embeddings = np.empty(
            (self._vocabulary_size, self.activation_width()), dtype=np.float32
        )
        getter.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        getter.restype = None
        getter(
            self._model._model.model,
            embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if not np.all(np.isfinite(embeddings)):
            raise RuntimeError("llama.cpp returned non-finite token embeddings")
        self._token_embedding_matrix_cache = embeddings
        return embeddings

    def activation_logit_adjustments(
        self,
        vector,
        *,
        layer: str = "output",
        position: str = "current",
        digest: str | None = None,
    ) -> np.ndarray:
        """Project a final-hidden activation delta through the GGUF output head.

        llama.cpp currently exposes the token embedding matrix, which is the
        output head for the tied-output models supported by this adapter.

        ``digest`` is the caller-known content digest of ``vector`` (the
        steering artifact digest the sampler config already carries).  When it
        is supplied the vector is not re-hashed on the interactive path; when
        it is absent the digest is derived from the vector bytes as before.
        """
        if layer != "output":
            raise RuntimeError("llama.cpp activation runtime currently supports layer=output only")
        if position != "current":
            raise RuntimeError("llama.cpp activation runtime position must be current")
        values = np.asarray(vector, dtype=np.float32)
        if values.ndim != 1 or values.shape[0] != self.activation_width():
            raise RuntimeError("output-head steering vector does not match the output-head width")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("output-head steering vector is not finite")
        cache_digest = digest or hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        key = (layer, position, cache_digest)
        cached = self._activation_logit_cache.get(key)
        if cached is not None:
            return cached.copy()
        result = self._token_embedding_matrix() @ values
        result = np.asarray(result, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise RuntimeError("llama.cpp activation output-head projection is not finite")
        self._activation_logit_cache[key] = result.copy()
        return result

    def close(self) -> None:
        """Release the inference context."""
        close = getattr(self._model, "close", None)
        if callable(close):
            close()
        self._model = None

    def tokenize(
        self, text: str, *, add_bos: bool = False, special: bool = False
    ) -> list[int]:
        return [
            int(value)
            for value in self._model.tokenize(
                text.encode("utf-8"), add_bos=add_bos, special=special
            )
        ]

    def new_text_stream(self, *, special: bool = False):
        return _LlamaCppTextStream(self._model, special=special)

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

    def tokenizer_id(self) -> str:
        """Return the tokenizer identity computed during backend setup."""

        return self._tokenizer_id_cache

    def model_id(self) -> str:
        return self._model_sha256()

    def _model_sha256(self) -> str:
        if self._model_sha256_cache is None:
            try:
                self._model_sha256_cache = sha256_path(self.model_path)
            except OSError as exc:
                raise RuntimeError(f"could not hash llama.cpp model: {exc}") from exc
        return self._model_sha256_cache

    def provenance(self, *, include_model_sha256: bool = True) -> dict[str, Any]:
        # Provenance identifies the model/runtime needed to interpret the run;
        # private evaluation state is neither evidence nor episode identity.
        stat = self.model_path.stat()
        requested = asdict(self.settings)
        if requested.get("tensor_split") is not None:
            requested["tensor_split"] = list(requested["tensor_split"])
        model_type = None
        meta_reader = getattr(self._llama_cpp, "llama_model_meta_val_str", None)
        if callable(meta_reader):
            buffer = ctypes.create_string_buffer(128)
            try:
                if int(meta_reader(self._model._model.model, b"general.architecture", buffer, 128)) >= 0:
                    model_type = buffer.value.decode("utf-8")
            except (TypeError, ValueError, UnicodeDecodeError):
                model_type = None
        result = {
            "backend": "llama.cpp",
            "adapter": "llama-cpp-python",
            "model_path": str(self.model_path.resolve()),
            "filename": self.model_path.name,
            "file_size_bytes": int(stat.st_size),
            "vocabulary_size": self.vocabulary_size(),
            "model_type": model_type,
            "activation_width": self.activation_width(),
            "activation_layer_count": self.activation_control_vector_layer_count(),
            "hidden_state_width": self.activation_control_vector_width(),
            "hidden_state_layer_count": self.activation_control_vector_layer_count(),
            "llama_cpp_python_version": getattr(self._llama_cpp, "__version__", None),
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "tokenizer_id": self.tokenizer_id(),
            "runtime_configuration": requested,
        }
        if include_model_sha256:
            result["model_sha256"] = self._model_sha256()
            result["model_id"] = result["model_sha256"]
        return result
