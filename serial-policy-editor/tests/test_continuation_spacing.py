from unittest.mock import patch

import pytest

from tests.test_episode_runtime import engine
from trajectory_editor.episode_actions import Write
from trajectory_editor.episode_cli import build_parser


@pytest.mark.parametrize("prior,text,continuation", [
    ("a", "nother", " nother"),
    ("a", "word", " word"),
    ("a ", "word", "word"),
    ("(", "word", "word"),
    ("a", " word", " word"),
    ("a", ",", ","),
    ("a", "\nword", "\nword"),
])
@pytest.mark.parametrize("replay", [False, True])
def test_write_modes_have_fixed_spacing_semantics(prior, text, continuation, replay):
    for mode, expected in [("continuation", continuation), ("exact", text)]:
        runtime = engine()
        with patch.object(runtime.backend, "render", return_value=prior), patch.object(
            runtime.backend, "tokenize", return_value=[4]
        ) as tokenize:
            outcome = runtime.apply(Write(text, mode), replay=replay)
        tokenize.assert_called_once_with(expected, add_bos=False, special=False)
        assert outcome.resolved_text == expected
        assert outcome.visible_token_ids == (4,)


def test_cli_no_longer_accepts_spacing_toggle():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--no-auto-space"])
