"""Hugging Face Transformers backend for the policy editor."""

from __future__ import annotations

import importlib.util
import inspect
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .domain import EditorError
from .episode_backend import CacheMode, validate_cache_mode


@dataclass(frozen=True)
class TransformersSettings:
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    device_map: str | None = None
    attention_implementation: str | None = None
    quantization_method: str = "none"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False
    torch_num_threads: int | None = None
    torch_num_interop_threads: int | None = None
    deterministic_algorithms: bool | None = None
    float32_matmul_precision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise EditorError("Transformers device must be a nonempty string")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise EditorError(
                "Transformers dtype must be auto, float32, float16, or bfloat16"
            )
        if self.device_map not in {
            None,
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        }:
            raise EditorError(
                "Transformers device map must be auto, balanced, balanced_low_0, or sequential"
            )
        if self.attention_implementation not in {
            None,
            "eager",
            "sdpa",
            "flash_attention_2",
            "flex_attention",
        }:
            raise EditorError("unsupported Transformers attention implementation")
        if self.quantization_method not in {
            "none",
            "bitsandbytes-8bit",
            "bitsandbytes-4bit",
        }:
            raise EditorError(
                "Transformers quantization must be none, bitsandbytes-8bit, or bitsandbytes-4bit"
            )
        if self.bnb_4bit_compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise EditorError(
                "bitsandbytes 4-bit compute dtype must be float32, float16, or bfloat16"
            )
        if self.bnb_4bit_quant_type not in {"fp4", "nf4"}:
            raise EditorError("bitsandbytes 4-bit quantization type must be fp4 or nf4")
        for value, label in (
            (self.torch_num_threads, "torch thread count"),
            (self.torch_num_interop_threads, "torch interop thread count"),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise EditorError(f"Transformers {label} must be a positive integer")
        if self.deterministic_algorithms is not None and type(
            self.deterministic_algorithms
        ) is not bool:
            raise EditorError(
                "Transformers deterministic algorithms setting must be a boolean"
            )
        if self.float32_matmul_precision not in {None, "highest", "high", "medium"}:
            raise EditorError(
                "Transformers float32 matmul precision must be highest, high, or medium"
            )


def _as_token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if type(value) is int:
        return {int(value)} if int(value) >= 0 else set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value if type(v) is int and int(v) >= 0}
    return set()


def _infer_context_limit(model_config: Any, tokenizer: Any) -> int | None:
    for name in (
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
    ):
        value = getattr(model_config, name, None)
        if type(value) is int and 0 < value < 100_000_000:
            return int(value)
    value = getattr(tokenizer, "model_max_length", None)
    if type(value) is int and 0 < value < 100_000_000:
        return int(value)
    return None


def _addressable_token_ids(tokenizer: Any) -> tuple[int, ...]:
    ids: set[int] = set()
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocabulary = get_vocab()
        if not isinstance(vocabulary, dict):
            raise RuntimeError("Transformers tokenizer.get_vocab() did not return a mapping")
        for raw_id in vocabulary.values():
            if type(raw_id) is not int or int(raw_id) < 0:
                raise RuntimeError("Transformers tokenizer vocabulary has an invalid token id")
            ids.add(int(raw_id))
    for raw_id in getattr(tokenizer, "all_special_ids", None) or []:
        if type(raw_id) is int and int(raw_id) >= 0:
            ids.add(int(raw_id))
    if not ids:
        length = int(len(tokenizer))
        if length <= 0:
            raise RuntimeError("Transformers tokenizer has no addressable token ids")
        ids.update(range(length))
    ordered = tuple(sorted(ids))
    expected = tuple(range(ordered[-1] + 1))
    if ordered != expected:
        raise RuntimeError(
            "Transformers tokenizer uses a sparse token-id domain; SPE currently "
            "requires contiguous ids 0..N"
        )
    return ordered


class _CacheUnavailable(RuntimeError):
    """The model cannot provide the optional incremental evaluation path."""


def _cache_length(cache: Any) -> int:
    get_length = getattr(cache, "get_seq_length", None)
    if callable(get_length):
        try:
            return int(get_length())
        except TypeError:
            return int(get_length(0))
    if isinstance(cache, (tuple, list)) and cache:
        first_layer = cache[0]
        if isinstance(first_layer, (tuple, list)) and first_layer:
            return int(first_layer[0].shape[-2])
    raise _CacheUnavailable("Transformers cache has no readable sequence length")


def _crop_cache(cache: Any, target_length: int) -> Any:
    current_length = _cache_length(cache)
    if target_length < 0 or target_length > current_length:
        raise _CacheUnavailable("Transformers cache crop target is out of range")
    tokens_to_remove = current_length - target_length
    if tokens_to_remove == 0:
        return cache
    crop = getattr(cache, "crop", None)
    if callable(crop):
        try:
            parameter_name = next(iter(inspect.signature(crop).parameters)).lower()
        except (TypeError, StopIteration, ValueError):
            parameter_name = "max_length"
        if "remove" in parameter_name:
            crop(-tokens_to_remove)
        else:
            crop(target_length)
        return cache
    if isinstance(cache, tuple):
        return tuple(
            tuple(value[..., :target_length, :].contiguous() for value in layer)
            for layer in cache
        )
    if isinstance(cache, list):
        return [
            tuple(value[..., :target_length, :].contiguous() for value in layer)
            for layer in cache
        ]
    raise _CacheUnavailable("installed Transformers cache cannot be cropped")


class TransformersBackend:
    """Local Hugging Face causal-LM adapter with optional KV reuse."""

    def __init__(
        self,
        model_path: Path,
        settings: TransformersSettings,
        *,
        cache_mode: CacheMode = "auto",
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch and Transformers are required; install this package with [transformers]"
            ) from exc

        self._torch = torch
        self._transformers = transformers
        self.model_path = Path(model_path)
        self.settings = settings
        self._cache_mode = validate_cache_mode(cache_mode)
        self._cache_enabled = self._cache_mode == "auto"
        self._cache_active = False
        self._past_key_values: Any | None = None
        if not self.model_path.is_dir():
            raise RuntimeError(
                "Transformers backend requires a local Hugging Face model directory: "
                f"{self.model_path}"
            )
        self._apply_execution_controls()

        common = {
            "local_files_only": True,
            "trust_remote_code": settings.trust_remote_code,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), use_fast=settings.use_fast_tokenizer, **common
        )

        version_text = str(getattr(transformers, "__version__", "0"))
        try:
            transformers_major = int(version_text.split(".", 1)[0])
        except ValueError:
            transformers_major = 0
        dtype_kwarg = "dtype" if transformers_major >= 5 else "torch_dtype"
        dtype_value: Any = settings.dtype
        if settings.dtype != "auto":
            dtype_value = getattr(torch, settings.dtype)
        model_kwargs = dict(common)
        model_kwargs[dtype_kwarg] = dtype_value
        if settings.device_map is not None:
            model_kwargs["device_map"] = settings.device_map
        if settings.attention_implementation is not None:
            model_kwargs["attn_implementation"] = settings.attention_implementation
        quantization_config = self._build_quantization_config()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), **model_kwargs
        )
        self._model.eval()

        self._device = self._resolve_device(settings.device)
        if settings.device_map is None:
            try:
                self._model.to(self._device)
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"could not place Transformers model on {self._device}: {exc}"
                ) from exc
        self._input_device = self._infer_input_device()

        self._context_limit = _infer_context_limit(self._model.config, self._tokenizer)
        self._model_output_size = int(getattr(self._model.config, "vocab_size", 0))
        if self._model_output_size <= 0:
            raise RuntimeError("Transformers model has no positive config.vocab_size")
        self._addressable_token_ids = _addressable_token_ids(self._tokenizer)
        self._vocabulary_size = len(self._addressable_token_ids)
        if self._model_output_size < self._vocabulary_size:
            raise RuntimeError(
                "Transformers model output head is smaller than tokenizer vocabulary"
            )
        self._eog_ids, self._eog_source = self._discover_eog_ids()
        self._tokens: list[int] = []
        self._last_logits: np.ndarray | None = None

    def _apply_execution_controls(self) -> None:
        if self.settings.torch_num_threads is not None:
            self._torch.set_num_threads(self.settings.torch_num_threads)
        if self.settings.torch_num_interop_threads is not None:
            self._torch.set_num_interop_threads(self.settings.torch_num_interop_threads)
        if self.settings.deterministic_algorithms is not None:
            self._torch.use_deterministic_algorithms(
                self.settings.deterministic_algorithms
            )
        if self.settings.float32_matmul_precision is not None:
            self._torch.set_float32_matmul_precision(
                self.settings.float32_matmul_precision
            )

    def _resolve_device(self, requested: str) -> Any:
        torch = self._torch
        if requested.strip().lower() != "auto":
            return torch.device(requested)
        if bool(torch.cuda.is_available()):
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")

    def _infer_input_device(self) -> Any:
        try:
            return next(self._model.parameters()).device
        except (StopIteration, AttributeError):
            return self._device

    def _build_quantization_config(self) -> Any | None:
        if self.settings.quantization_method == "none":
            return None
        if importlib.util.find_spec("bitsandbytes") is None:
            raise EditorError(
                "bitsandbytes quantization requested but bitsandbytes is not installed"
            )
        config_class = getattr(self._transformers, "BitsAndBytesConfig", None)
        if config_class is None:
            raise EditorError("installed Transformers has no BitsAndBytesConfig")
        if self.settings.quantization_method == "bitsandbytes-8bit":
            return config_class(load_in_8bit=True)
        return config_class(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                self._torch, self.settings.bnb_4bit_compute_dtype
            ),
            bnb_4bit_quant_type=self.settings.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.settings.bnb_4bit_use_double_quant,
        )

    def _discover_eog_ids(self) -> tuple[set[int], str]:
        generation = getattr(self._model, "generation_config", None)
        for source, value in (
            (
                "model.generation_config.eos_token_id",
                getattr(generation, "eos_token_id", None),
            ),
            ("model.config.eos_token_id", getattr(self._model.config, "eos_token_id", None)),
            ("tokenizer.eos_token_id", getattr(self._tokenizer, "eos_token_id", None)),
        ):
            ids = _as_token_id_set(value)
            if ids:
                invalid = [v for v in ids if v >= self._vocabulary_size]
                if invalid:
                    raise RuntimeError(f"{source} contains invalid token ids {invalid}")
                return ids, source
        return set(), "none"

    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    def _validate_tokens(self, values: list[int]) -> None:
        if any(v < 0 or v >= self._vocabulary_size for v in values):
            raise RuntimeError("prefix contains a token id outside the model vocabulary")
        if self._context_limit is not None and len(values) > self._context_limit:
            raise RuntimeError(
                f"Transformers prefix exceeds context limit: {len(values)} > {self._context_limit}"
            )

    def reset(self, prefix_token_ids: list[int]) -> None:
        if not prefix_token_ids:
            raise RuntimeError("decoder prefix cannot be empty")
        self._tokens = [int(v) for v in prefix_token_ids]
        self._past_key_values = None
        self._cache_active = False
        if self._cache_enabled:
            try:
                self._evaluate_complete_prefix(use_cache=True)
                return
            except (AttributeError, TypeError, ValueError, _CacheUnavailable):
                # Cache support is optional. Fall through to the canonical
                # complete-prefix path when this model/configuration lacks it.
                self._past_key_values = None
                self._cache_active = False
        self._evaluate_complete_prefix(use_cache=False)

    def eval(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        values = [int(v) for v in token_ids]
        if self._cache_active:
            try:
                self._evaluate_incremental(values)
                return
            except (AttributeError, TypeError, ValueError, _CacheUnavailable):
                self._past_key_values = None
                self._cache_active = False
        self._tokens.extend(values)
        self._evaluate_complete_prefix(use_cache=False)

    def _evaluate_complete_prefix(self, *, use_cache: bool) -> None:
        self._validate_tokens(self._tokens)
        torch = self._torch
        input_ids = torch.tensor(
            [self._tokens], dtype=torch.long, device=self._input_device
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                return_dict=True,
            )
        if use_cache:
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise _CacheUnavailable(
                    "Transformers model returned no past_key_values"
                )
            self._past_key_values = past_key_values
            self._cache_active = True
        else:
            self._past_key_values = None
            self._cache_active = False
        self._set_last_logits(getattr(outputs, "logits", None))

    def _evaluate_incremental(self, values: list[int]) -> None:
        all_values = [*self._tokens, *values]
        self._validate_tokens(all_values)
        torch = self._torch
        old_length = len(self._tokens)
        new_length = len(all_values)
        input_ids = torch.tensor([values], dtype=torch.long, device=self._input_device)
        attention_mask = torch.ones(
            (1, new_length), dtype=torch.long, device=self._input_device
        )
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": self._past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        cache_position = torch.arange(
            old_length, new_length, dtype=torch.long, device=self._input_device
        )
        try:
            with torch.inference_mode():
                outputs = self._model(**kwargs, cache_position=cache_position)
        except TypeError:
            # Older Transformers versions still support past-key-values but do
            # not accept the cache_position keyword.
            with torch.inference_mode():
                outputs = self._model(**kwargs)
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise _CacheUnavailable(
                "Transformers model returned no past_key_values during increment"
            )
        self._past_key_values = past_key_values
        self._set_last_logits(getattr(outputs, "logits", None))
        self._tokens = all_values

    def branch_to_prefix(self, prefix_token_ids: list[int]) -> None:
        """Move to an existing prefix, reusing and cropping the current cache."""
        values = [int(v) for v in prefix_token_ids]
        if not values:
            raise RuntimeError("decoder prefix cannot be empty")
        if not self._cache_active:
            self.reset(values)
            return
        if len(values) > len(self._tokens) or self._tokens[: len(values)] != values:
            self.reset(values)
            return
        if len(values) == len(self._tokens):
            return
        try:
            retained_before_last = len(values) - 1
            self._past_key_values = _crop_cache(
                self._past_key_values, retained_before_last
            )
            self._tokens = values[:-1]
            self._evaluate_incremental([values[-1]])
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # Cache state is an optimization. Rebuild the requested prefix if
            # the installed model/cache implementation cannot be repositioned.
            self.reset(values)

    def _set_last_logits(self, logits: Any) -> None:
        if logits is None or getattr(logits, "ndim", None) != 3:
            raise RuntimeError("Transformers causal LM returned no 3-D logits tensor")
        if int(logits.shape[0]) != 1 or int(logits.shape[2]) < self._vocabulary_size:
            raise RuntimeError("Transformers logits shape does not cover tokenizer vocabulary")
        row = (
            logits[0, -1, : self._vocabulary_size]
            .detach()
            .to(dtype=self._torch.float32, device="cpu")
            .numpy()
        )
        result = np.asarray(row, dtype=np.float32).copy()
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Transformers returned non-finite logits")
        self._last_logits = result

    def last_logits(self) -> np.ndarray:
        if self._last_logits is None:
            raise RuntimeError("decoder has not evaluated a prefix")
        return self._last_logits.copy()

    def tokenize(
        self, text: str, *, add_bos: bool = False, special: bool = False
    ) -> list[int]:
        token_ids = [
            int(v)
            for v in self._tokenizer.encode(text, add_special_tokens=False)
        ]
        if not special:
            registered_special = {
                int(v)
                for v in (getattr(self._tokenizer, "all_special_ids", None) or [])
                if type(v) is int and int(v) >= 0
            }
            encountered = sorted(set(token_ids) & registered_special)
            if encountered:
                raise RuntimeError(
                    "literal tokenization resolved to registered special token id(s) "
                    f"{encountered}"
                )
        if add_bos:
            bos = getattr(self._tokenizer, "bos_token_id", None)
            if type(bos) is int and int(bos) >= 0:
                token_ids.insert(0, int(bos))
        return token_ids

    def render(self, token_ids: list[int], *, special: bool = False) -> str:
        if not token_ids:
            return ""
        values = [int(v) for v in token_ids]
        try:
            return str(
                self._tokenizer.decode(
                    values,
                    skip_special_tokens=not special,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(
                self._tokenizer.decode(values, skip_special_tokens=not special)
            )

    def token_text(self, token_id: int) -> str:
        token_id = int(token_id)
        text = self.render([token_id], special=self.is_eog(token_id))
        return text or ("<EOG>" if self.is_eog(token_id) else "")

    def is_eog(self, token_id: int) -> bool:
        return int(token_id) in self._eog_ids

    def eog_token_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._eog_ids))

    def provenance(self, *, include_model_sha256: bool = True) -> dict[str, Any]:
        config = self._model.config
        try:
            loaded_dtype = str(next(self._model.parameters()).dtype)
        except (StopIteration, AttributeError):
            loaded_dtype = None
        return {
            "backend": "transformers",
            "adapter": "huggingface-transformers-causal-lm",
            "model_path": str(self.model_path.resolve()),
            "vocabulary_size": self.vocabulary_size(),
            "context_limit": self._context_limit,
            "eog_token_ids": list(self.eog_token_ids()),
            "eog_source": self._eog_source,
            "model_type": getattr(config, "model_type", None),
            "transformers_version": getattr(self._transformers, "__version__", None),
            "torch_version": getattr(self._torch, "__version__", None),
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "runtime_configuration": {
                **asdict(self.settings),
                "effective_device": str(self._input_device),
                "loaded_dtype": loaded_dtype,
            },
        }
