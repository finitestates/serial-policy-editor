"""Portable, versioned single-token bias presets (token IDs are model-specific)."""
from __future__ import annotations

import json
from pathlib import Path

from .domain import EditorError, SamplingConfig

FORMAT = "spe-logit-bias-v1"
IDENTITY_FIELDS = ("backend", "filename", "file_size_bytes", "vocabulary_size")


def load_bias_preset(path: Path, backend, provenance: dict) -> tuple[tuple[int, float], ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EditorError(f"Could not read bias preset: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise EditorError(f"Bias preset must use format {FORMAT}")
    model = value.get("model")
    rows = value.get("biases")
    if not isinstance(model, dict) or not isinstance(rows, list):
        raise EditorError("Bias preset requires model metadata and a biases list")
    if type(model.get("vocabulary_size")) is not int or model["vocabulary_size"] != backend.vocabulary_size():
        raise EditorError("Bias preset vocabulary does not match the loaded model")
    for field in IDENTITY_FIELDS:
        if field in model and model[field] != provenance.get(field):
            raise EditorError(f"Bias preset model mismatch: {field}")
    pairs = []
    for row in rows:
        if not isinstance(row, dict) or not {"token_id", "bias"} <= row.keys():
            raise EditorError("Each bias requires token_id and bias")
        pairs.append((row["token_id"], row["bias"]))
    result = SamplingConfig(logit_bias=pairs).logit_bias
    for token, _ in pairs:
        if token >= backend.vocabulary_size():
            raise EditorError("Bias token ID is outside the loaded vocabulary")
    for row in rows:
        if "text" in row and row["text"] != backend.token_text(row["token_id"]):
            raise EditorError(f"Bias preset token text mismatch at ID {row['token_id']}")
    return result


def project_biases(store, episode_id: str) -> str:
    episode = store.get_episode(episode_id)
    config = store.final_sampling(episode_id)
    labels = {}
    for row in store.interactions(episode_id):
        if row["kind"] == "logit-bias":
            payload = row["payload"]
            labels[payload["token_id"]] = payload["text"]
    model = {key: episode["backend"][key] for key in IDENTITY_FIELDS
             if episode["backend"].get(key) is not None}
    rows = [{"token_id": token, "bias": bias,
             **({"text": labels[token]} if token in labels else {})}
            for token, bias in config.logit_bias]
    return json.dumps({"format": FORMAT, "model": model, "biases": rows},
                      ensure_ascii=False, indent=2, allow_nan=False)
