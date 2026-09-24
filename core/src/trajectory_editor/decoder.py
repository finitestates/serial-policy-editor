"""llama-cpp-python adapter for the policy editor."""

from __future__ import annotations

from contextlib import nullcontext

import ctypes
import hashlib
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core.errors import EditorError
from .core.backend import BackendStateSnapshot, CacheMode, InferenceBackend, validate_cache_mode
from .model_hash import sha256_path


# Public spelling retained because backend_factory and a few user scripts used it.
Decoder = InferenceBackend

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
    *,
    embedding: bool = False,
) -> dict[str, Any]:
    """Build shared options for inference and the auxiliary embedding context."""
    options = {
        "model_path": str(model_path),
        "n_ctx": settings.n_ctx,
        "n_batch": settings.n_batch,
        "flash_attn": settings.flash_attn,
        "use_mmap": settings.use_mmap,
        "use_mlock": settings.use_mlock,
        "logits_all": False,
        "embedding": embedding,
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
        self._snapshot_token = object()
        self._token_embedding_matrix_cache: np.ndarray | None = None
        self._activation_model: Any | None = None
        self._activation_logit_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self._model_sha256_cache: str | None = None
        # Set only by the optional real-model harness. Normal inference has no
        # measurement object or extra synchronization.
        self._real_model_probe = None

    def _measured_eval(self, values: list[int], kind: str, context_length: int) -> None:
        probe = getattr(self, "_real_model_probe", None)
        interval = probe.model_call(kind, len(values), context_length) if probe is not None else nullcontext()
        with interval:
            self._model.eval(values)

    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def reset(self, prefix_token_ids: list[int]) -> None:
        if not prefix_token_ids:
            raise RuntimeError("decoder prefix cannot be empty")
        values = [int(value) for value in prefix_token_ids]
        self._model.reset()
        self._measured_eval(values, "prefill", len(values))
        self._tokens = values

    def eval(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        values = [int(value) for value in token_ids]
        self._tokens.extend(values)
        if self._cache_enabled:
            self._measured_eval(values, "incremental", len(self._tokens))
        else:
            self._model.reset()
            self._measured_eval(self._tokens, "rebuild", len(self._tokens))

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
                probe = getattr(self, "_real_model_probe", None)
                if probe is not None:
                    probe.cache_fallback("branch-cache-remove-rejected")
                self.reset(values)
                return
            self._model.n_tokens = retained_before_last
            self._model._requires_eval = True
            self._measured_eval([values[-1]], "branch", len(values))
            self._tokens = values
        except (AttributeError, RuntimeError, TypeError):
            # Cache positioning is an optimization. If the installed binding
            # cannot safely truncate its sequence, preserve semantics via reset.
            probe = getattr(self, "_real_model_probe", None)
            if probe is not None:
                probe.cache_fallback("branch-cache-unavailable")
            self.reset(values)

    def snapshot_state(self) -> BackendStateSnapshot | None:
        """Capture llama.cpp's full context state when the wrapper supports it."""
        save_state = getattr(self._model, "save_state", None)
        if not callable(save_state) or not self._tokens:
            return None
        try:
            state = save_state()
        except Exception:
            return None
        return BackendStateSnapshot(
            self._snapshot_token, tuple(self._tokens), state
        )

    def restore_state(self, snapshot: BackendStateSnapshot) -> bool:
        """Restore an exact llama.cpp state saved by this backend instance."""
        if (
            not isinstance(snapshot, BackendStateSnapshot)
            or snapshot.backend_token is not self._snapshot_token
        ):
            return False
        load_state = getattr(self._model, "load_state", None)
        if not callable(load_state):
            return False
        # Let native restoration failures surface: the context may be partially
        # repositioned, so callers must rebuild from the semantic prefix.
        load_state(snapshot.payload)
        self._tokens = list(snapshot.prefix_token_ids)
        return True

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

    def hidden_state_width(self) -> int:
        """Return the residual-stream width used by native control vectors."""
        return self.activation_width()

    def hidden_state_layer_count(self) -> int:
        """Return the number of canonical decoder block-output coordinates."""
        return self._control_vector_layer_count()

    def hidden_state_capture_layer_range(self) -> tuple[int, int]:
        """Return the block outputs exposed by llama.cpp's layer taps."""
        return 1, max(1, self._model_layer_count() - 1)

    def hidden_state_runtime_layer_range(self) -> tuple[int, int]:
        """Return canonical block-output layers supported by native injection."""
        layer_count = self.hidden_state_layer_count()
        if layer_count < 2:
            raise RuntimeError(
                "llama.cpp control vectors expose no canonical block-output runtime layer"
            )
        return 2, layer_count

    def hidden_state_layer_types(self) -> tuple[str, ...]:
        return tuple("decoder" for _ in range(self.hidden_state_layer_count()))

    def hidden_state_capabilities(self) -> dict[str, Any]:
        """Describe llama.cpp's native residual-stream capture coordinate."""
        return {
            "site": "decoder-block-output-residual",
            "layer_numbering": "one-based",
            "layer_count": self.hidden_state_layer_count(),
            "width": self.hidden_state_width(),
            "position_policies": ["first", "last", "current", "all"],
            "layer_types": list(self.hidden_state_layer_types()),
            "native_module_path": "llama.cpp layer input tap N",
            "capture_coordinate": "canonical block-output N <- native input tap N",
            "injection_coordinate": "canonical block-output N -> native cvector slot N-1",
            "capture_layer_range": list(self.hidden_state_capture_layer_range()),
            "runtime_layer_range": list(self.hidden_state_runtime_layer_range()),
            "modality": "text",
            "final_layer_policy": "worker-graph-output-callback",
        }

    def _hidden_state_capture_symbols(self):
        """Resolve llama.cpp's internal per-layer extraction extension.

        These functions are currently exported by llama.cpp's extension header
        rather than the stable public header. The local build exposes their
        C++ symbols; accepting public names as well keeps this compatible with
        a future C-ABI promotion.
        """
        binding = getattr(self._llama_cpp, "llama_cpp", self._llama_cpp)
        library = getattr(binding, "_lib", None)
        if library is None:
            raise RuntimeError(
                "installed llama-cpp-python does not expose the native layer-capture library"
            )

        def resolve(public_name: str, mangled_name: str, restype, argtypes):
            function = getattr(self._llama_cpp, public_name, None)
            if not callable(function):
                function = getattr(library, public_name, None)
            if function is None:
                function = getattr(library, mangled_name, None)
            if function is None:
                raise RuntimeError(
                    "installed llama.cpp build does not expose " + public_name
                )
            function.argtypes = argtypes
            function.restype = restype
            return function

        context_type = ctypes.c_void_p
        setter = resolve(
            "llama_set_embeddings_layer_inp",
            "_Z30llama_set_embeddings_layer_inpP13llama_contextjb",
            None,
            [context_type, ctypes.c_uint32, ctypes.c_bool],
        )
        getter = resolve(
            "llama_get_embeddings_layer_inp",
            "_Z30llama_get_embeddings_layer_inpP13llama_contextj",
            ctypes.POINTER(ctypes.c_float),
            [context_type, ctypes.c_uint32],
        )
        return setter, getter

    def _capture_hidden_state_rows(
        self, text: str, layers: tuple[int, ...]
    ) -> dict[int, np.ndarray]:
        """Capture all token rows for selected residual layers in an isolated context."""
        if not layers:
            raise RuntimeError("at least one hidden-state layer must be selected")
        capture_start, capture_end = self.hidden_state_capture_layer_range()
        if any(
            type(layer) is not int or layer < capture_start or layer > capture_end
            for layer in layers
        ):
            raise RuntimeError(
                f"hidden-state layer must be between {capture_start} and {capture_end}"
            )
        if not isinstance(text, str) or not text:
            raise RuntimeError("hidden-state snapshot prompt must be nonempty")

        setter, getter = self._hidden_state_capture_symbols()
        model = self._activation_embedding_model()
        tokens = model.tokenize(text.encode("utf-8"), add_bos=True, special=False)
        if not tokens:
            raise RuntimeError("hidden-state snapshot prompt produced no tokens")
        if len(tokens) > int(model.n_batch):
            raise RuntimeError(
                f"hidden-state snapshot has {len(tokens)} tokens; "
                f"the capture batch supports {int(model.n_batch)}"
            )

        context = model._ctx.ctx if hasattr(model, "_ctx") else model.ctx
        width = int(model.n_embd())
        enabled: list[int] = []
        try:
            for layer in layers:
                setter(context, layer, True)
                enabled.append(layer)

            model._batch.reset()
            model._ctx.kv_cache_clear()
            model._batch.add_sequence(tokens, 0, True)
            result = model._ctx.decode(model._batch)
            if result not in (None, 0):
                raise RuntimeError(
                    f"llama.cpp hidden-state capture failed (error {result})"
                )

            captured: dict[int, np.ndarray] = {}
            for layer in layers:
                pointer = getter(context, layer)
                if not pointer:
                    raise RuntimeError(
                        f"llama.cpp returned no hidden-state capture for layer {layer}"
                    )
                values = np.ctypeslib.as_array(
                    pointer, shape=(len(tokens) * width,)
                ).astype(np.float32, copy=True)
                values = values.reshape(len(tokens), width)
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(
                        f"llama.cpp hidden-state capture for layer {layer} is not finite"
                    )
                captured[layer] = values
            return captured
        finally:
            for layer in enabled:
                try:
                    setter(context, layer, False)
                except (OSError, RuntimeError, TypeError):
                    pass
            try:
                model._batch.reset()
                model._ctx.kv_cache_clear()
                model.reset()
            except (AttributeError, RuntimeError, TypeError):
                pass

    def hidden_state_snapshot(
        self,
        text: str,
        *,
        layer: int,
        position: str = "last",
    ) -> np.ndarray:
        """Capture a residual-stream state at a native llama.cpp layer."""
        if position not in {"first", "last", "current", "all"}:
            raise RuntimeError(
                "hidden-state snapshot position must be first, last, current, or all"
            )
        states = self._capture_hidden_state_rows(text, (layer,))[layer]
        if position == "all":
            return states
        return np.asarray(
            states[0 if position == "first" else -1], dtype=np.float32
        ).copy()

    def hidden_state_snapshots(
        self,
        text: str,
        *,
        layer_start: int,
        layer_end: int,
        position: str = "last",
    ) -> dict[int, np.ndarray]:
        """Capture selected residual layers with one llama.cpp evaluation."""
        if position not in {"first", "last", "current", "all"}:
            raise RuntimeError(
                "hidden-state snapshot position must be first, last, current, or all"
            )
        capture_start, capture_end = self.hidden_state_capture_layer_range()
        if (
            type(layer_start) is not int
            or type(layer_end) is not int
            or layer_start < capture_start
            or layer_end < layer_start
            or layer_end > capture_end
        ):
            raise RuntimeError(
                f"hidden-state layer range must be between {capture_start} and {capture_end}"
            )
        rows = self._capture_hidden_state_rows(
            text, tuple(range(layer_start, layer_end + 1))
        )
        if position == "all":
            return rows
        index = 0 if position == "first" else -1
        return {
            layer: np.asarray(values[index], dtype=np.float32).copy()
            for layer, values in rows.items()
        }

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
        native slot ``N - 1``.  Canonical layer 1 remains capture-only because
        the upstream native adapter reserves no slot 0; canonical layer N is
        fully steerable before final output normalization.
        """
        setter = getattr(self._llama_cpp, "llama_set_adapter_cvec", None)
        if not callable(setter):
            raise RuntimeError("installed llama.cpp binding does not expose control vectors")
        width = self.activation_control_vector_width()
        layer_count = self.activation_control_vector_layer_count()
        runtime_start, runtime_end = self.hidden_state_runtime_layer_range()
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
        # Native llama.cpp cvector data begins at slot 1, so discard the
        # capture-only canonical layer 1 chunk and pass canonical layers 2..N
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

    def _activation_embedding_model(self) -> Any:
        if self._activation_model is None:
            self._activation_model = self._model.__class__(
                **_llama_options(
                    self._llama_cpp,
                    self.model_path,
                    self.settings,
                    embedding=True,
                )
            )
        return self._activation_model

    def activation_snapshot(
        self,
        text: str,
        *,
        layer: str = "output",
        position: str = "last",
    ) -> np.ndarray:
        """Capture one final hidden-state position for a prompt.

        llama.cpp exposes the final representation through its embedding
        context. Internal residual capture is provided separately by
        ``hidden_state_snapshot``.
        """
        if layer != "output":
            raise RuntimeError("llama.cpp activation snapshots currently support layer=output only")
        if position not in {"first", "last"}:
            raise RuntimeError("activation snapshot position must be first or last")
        if not isinstance(text, str) or not text:
            raise RuntimeError("activation snapshot prompt must be nonempty")
        model = self._activation_embedding_model()
        values = np.asarray(
            model.embed(text, normalize=False, truncate=False), dtype=np.float32
        )
        if values.ndim != 2 or values.shape[0] < 1:
            raise RuntimeError("llama.cpp returned no per-token activation snapshot")
        row = values[0 if position == "first" else -1].copy()
        if row.ndim != 1 or row.shape[0] != self.activation_width():
            raise RuntimeError("llama.cpp activation snapshot has the wrong width")
        if not np.all(np.isfinite(row)):
            raise RuntimeError("llama.cpp activation snapshot is not finite")
        return row

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
    ) -> np.ndarray:
        """Project a final-hidden activation delta through the GGUF output head.

        llama.cpp currently exposes the token embedding matrix, which is the
        output head for the tied-output models supported by this adapter.
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
        digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        key = (layer, position, digest)
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
        """Release both the normal and auxiliary activation contexts."""
        for name in ("_activation_model", "_model"):
            model = getattr(self, name, None)
            if model is not None:
                close = getattr(model, "close", None)
                if callable(close):
                    close()
                setattr(self, name, None)

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
            "hidden_state_width": self.hidden_state_width(),
            "hidden_state_layer_count": self.hidden_state_layer_count(),
            "llama_cpp_python_version": getattr(self._llama_cpp, "__version__", None),
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "runtime_configuration": requested,
        }
        if include_model_sha256:
            result["model_sha256"] = self._model_sha256()
        return result
