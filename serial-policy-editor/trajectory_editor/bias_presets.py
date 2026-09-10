"""Portable, versioned token and sequence bias presets (token IDs are model-specific)."""
from __future__ import annotations

import json
from pathlib import Path

from .domain import EditorError, SamplingConfig

FORMAT = "spe-logit-bias-v1"
SEQUENCE_FORMAT = "spe-logit-bias-v2"
IDENTITY_FIELDS = ("backend", "filename", "file_size_bytes", "vocabulary_size")


def load_bias_preset(path: Path, backend, provenance: dict) -> SamplingConfig:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EditorError(f"Could not read bias preset: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") not in {FORMAT, SEQUENCE_FORMAT}:
        raise EditorError(f"Bias preset must use format {FORMAT} or {SEQUENCE_FORMAT}")
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
    sequences = []
    addressed = []
    for row in rows:
        if not isinstance(row, dict) or "bias" not in row:
            raise EditorError("Each bias requires token IDs and bias")
        if value["format"] == SEQUENCE_FORMAT:
            tokens = row.get("token_ids")
            if not isinstance(tokens, list) or not tokens:
                raise EditorError("Each v2 bias requires a nonempty token_ids list")
            sequences.append((tokens, row["bias"]))
        else:
            if "token_id" not in row:
                raise EditorError("Each v1 bias requires token_id and bias")
            tokens = [row["token_id"]]
            pairs.append((tokens[0], row["bias"]))
        addressed.append((tokens, row))
    result = SamplingConfig(logit_bias=pairs, sequence_bias=sequences)
    for tokens, row in addressed:
        if any(token >= backend.vocabulary_size() for token in tokens):
            raise EditorError("Bias token ID is outside the loaded vocabulary")
        if "texts" in row and row["texts"] != [backend.token_text(token) for token in tokens]:
            raise EditorError("Bias preset token text mismatch")
        if "text" in row and (len(tokens) != 1 or row["text"] != backend.token_text(tokens[0])):
            raise EditorError(f"Bias preset token text mismatch at ID {tokens[0]}")
    return result


def project_biases(store, episode_id: str) -> str:
    episode = store.get_episode(episode_id)
    config = store.final_sampling(episode_id)
    labels = {}
    sequence_labels = {}
    for row in store.interactions(episode_id):
        if row["kind"] == "logit-bias":
            payload = row["payload"]
            labels[payload["token_id"]] = payload["text"]
        elif row["kind"] == "sequence-bias":
            payload = row["payload"]
            sequence_labels[tuple(payload["token_ids"])] = payload["texts"]
    model = {key: episode["backend"][key] for key in IDENTITY_FIELDS
             if episode["backend"].get(key) is not None}
    rows = [{"token_id": token, "bias": bias,
             **({"text": labels[token]} if token in labels else {})}
            for token, bias in config.logit_bias]
    format_name = FORMAT
    if config.sequence_bias:
        format_name = SEQUENCE_FORMAT
        rows = [{"token_ids": [row["token_id"]], "bias": row["bias"],
                 **({"texts": [row["text"]]} if "text" in row else {})} for row in rows]
        rows.extend({"token_ids": list(tokens), "bias": bias,
                     **({"texts": sequence_labels[tokens]} if tokens in sequence_labels else {})}
                    for tokens, bias in config.sequence_bias)
    return json.dumps({"format": format_name, "model": model, "biases": rows},
                      ensure_ascii=False, indent=2, allow_nan=False)
