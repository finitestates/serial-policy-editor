from __future__ import annotations

from typing import Any

import numpy as np


class ConformingFakeBackend:
    pieces = {
        0: "<EOG>",
        1: " A",
        2: " B",
        3: " C",
        4: " hello",
        5: "!",
        6: "?",
        7: "P",
    }

    def __init__(self) -> None:
        self.tokens: list[int] = []

    def reset(self, prefix_token_ids: list[int]) -> None:
        self.tokens = list(prefix_token_ids)

    def vocabulary_size(self) -> int:
        return len(self.pieces)

    def eval(self, token_ids: list[int]) -> None:
        self.tokens.extend(int(v) for v in token_ids)

    def last_logits(self) -> np.ndarray:
        logits = np.full(self.vocabulary_size(), -10.0, dtype=np.float32)
        last_token = self.tokens[-1] if self.tokens else None
        if last_token == 7:
            logits[1], logits[2], logits[3], logits[0] = 10.0, 8.0, 7.0, -9.0
        elif last_token in {1, 4}:
            logits[2], logits[3], logits[1], logits[0] = 10.0, 8.0, 7.0, -9.0
        else:
            logits[0], logits[5], logits[6] = 10.0, 0.0, -1.0
        return logits

    def tokenize(
        self, text: str, *, add_bos: bool = False, special: bool = False
    ) -> list[int]:
        del special
        if add_bos:
            return [7]
        mapping = {
            " hello": [4],
            "hello": [4],
            "C": [3],
            " P": [7],
            "P": [7],
            " A": [1],
            " B": [2],
            " A B": [1, 2],
            "!": [5],
            "?": [6],
        }
        return list(mapping.get(text, [6] if text else []))

    def render(self, token_ids: list[int], *, special: bool = False) -> str:
        del special
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)

    def token_text(self, token_id: int) -> str:
        return self.pieces[int(token_id)]

    def is_eog(self, token_id: int) -> bool:
        return int(token_id) == 0

    def eog_token_ids(self) -> tuple[int, ...]:
        return (0,)

    def provenance(self, *, include_model_sha256: bool = True) -> dict[str, Any]:
        del include_model_sha256
        return {"backend": "fake", "vocabulary_size": self.vocabulary_size()}


class ChangedProposalBackend(ConformingFakeBackend):
    def last_logits(self) -> np.ndarray:
        logits = super().last_logits()
        if self.tokens == [7]:
            logits[1], logits[2] = logits[2], logits[1]
        return logits


class BranchingFakeBackend(ConformingFakeBackend):
    """Fake backend that records whether fork positioning reuses its state."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_calls = 0
        self.eval_calls = 0
        self.branch_prefixes: list[list[int]] = []

    def reset(self, prefix_token_ids: list[int]) -> None:
        self.reset_calls += 1
        super().reset(prefix_token_ids)

    def eval(self, token_ids: list[int]) -> None:
        self.eval_calls += 1
        super().eval(token_ids)

    def branch_to_prefix(self, prefix_token_ids: list[int]) -> None:
        self.branch_prefixes.append(list(prefix_token_ids))
        self.tokens = list(prefix_token_ids)


class ScriptedIO:
    def __init__(self, responses: list[str | None]) -> None:
        self.responses = list(responses)
        self.output: list[str] = []

    @property
    def supports_live_choices(self) -> bool:
        return False

    def read(self, prompt: str) -> str | None:
        self.output.append(prompt)
        if not self.responses:
            raise AssertionError(f"unexpected input request: {prompt}")
        return self.responses.pop(0)

    def read_key(self, prompt: str) -> str | None:
        self.output.append(prompt)
        if not self.responses:
            raise AssertionError(f"unexpected key input request: {prompt}")
        return self.responses.pop(0)

    def write(self, text: str = "", *, end: str = "\n") -> None:
        self.output.append(text + end)

    def page(self, text: str) -> None:
        self.output.append(text)

    def progress(self, text: str, *, done: bool = False) -> None:
        self.output.append(text + ("\n" if done else ""))
