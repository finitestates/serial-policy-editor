"""Runtime observation seam with a lazy research compatibility adapter.

The normal path imports only the dependency-light core observer. Older
``SamplingConfig`` records that still carry research actuators are handed to
the legacy observer lazily, so the engine does not need to import or name
those actuators.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .core.errors import EditorError
from .core.observation import ObservationStatistics as CoreObservationStatistics


def _legacy_record(config: Any) -> bool:
    """Identify the wider saved configuration record without importing it."""

    return any(
        hasattr(config, name)
        for name in (
            "token_preference_vector",
            "group_controls",
            "reference_prior_routes",
        )
    )


def ObservationStatistics(*args: Any, **kwargs: Any) -> Any:
    """Compatibility factory preserving the historical observer import seam."""

    config = args[1] if len(args) > 1 else kwargs.get("config")
    if _legacy_record(config):
        from .sampling import ObservationStatistics as LegacyObservationStatistics

        return LegacyObservationStatistics(*args, **kwargs)
    return CoreObservationStatistics(*args, **kwargs)


def _load_preference_features(config: Any, backend: Any) -> dict[str, Any]:
    """Prepare old token-preference inputs without importing them on startup."""

    slow_vector = tuple(getattr(config, "token_preference_vector", ()))
    fast_vector = tuple(getattr(config, "token_preference_fast_vector", ()))
    if not (slow_vector or fast_vector):
        return {}
    provider = getattr(backend, "token_preference_features", None)
    if not callable(provider):
        raise EditorError(
            "the loaded backend does not expose token embeddings for token preference"
        )
    from .vector_artifacts import _supported_kwargs

    kwargs = dict(
        feature_dimension=len(slow_vector or fast_vector),
        projection_seed=config.token_preference_projection_seed,
        feature_scheme=config.token_preference_feature_scheme,
        whitening_ridge=config.token_preference_whitening_ridge,
    )
    try:
        features = provider(**_supported_kwargs(provider, kwargs))
        identity_method = getattr(backend, "token_preference_coordinate_identity", None)
        identity = (
            identity_method(**_supported_kwargs(identity_method, kwargs))
            if callable(identity_method)
            else None
        )
        if callable(identity_method):
            from .token_preference_features import coordinate_identity_matches

            if not coordinate_identity_matches(
                config,
                dimension=features.shape[1],
                model_fingerprint=identity.model_fingerprint,
                embedding_width=identity.embedding_width,
            ):
                raise EditorError(
                    "token preference coordinate system does not match the loaded model; "
                    "reset token preference memory before continuing"
                )
        return {
            "token_preference_features": features,
            "token_preference_coordinate_identity": identity,
        }
    except (TypeError, ValueError, RuntimeError) as exc:
        raise EditorError(f"could not load preference token features: {exc}") from exc


class ControllerPipeline:
    """Build policy observations through one trace-capable runtime seam."""

    def __init__(self, *, capture_trace: bool = False) -> None:
        self.capture_trace = bool(capture_trace)

    def prepare_sampling(self, config: Any, *, initial_token_count: int) -> Any:
        """Prepare a wider research record before core execution.

        The core engine deliberately does not know about group controls.  The
        compatibility adapter owns the historical rule that an unspecified
        control begins after the initial prompt.
        """

        group_controls = tuple(getattr(config, "group_controls", ()))
        if not group_controls:
            return config
        return replace(
            config,
            group_controls=tuple(
                replace(control, history_start=initial_token_count)
                if control.history_start is None else control
                for control in group_controls
            ),
        )

    def build_statistics(
        self,
        logits: Any,
        config: Any,
        history_token_ids: Any,
        boundaries: Any = None,
        *,
        backend: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Compatibility path only: this import brings in the research and
        # reference machinery for old records, never for the core runtime.
        if (
            _legacy_record(config)
            and kwargs.get("token_preference_features") is None
            and (
                getattr(config, "token_preference_vector", ())
                or getattr(config, "token_preference_fast_vector", ())
            )
        ):
            if backend is None:
                raise EditorError(
                    "a backend is required to load token preference features"
                )
            kwargs.update(_load_preference_features(config, backend))
        return ObservationStatistics(
            logits,
            config,
            history_token_ids,
            boundaries,
            capture_trace=self.capture_trace,
            **kwargs,
        )


__all__ = ["ControllerPipeline", "ObservationStatistics"]
