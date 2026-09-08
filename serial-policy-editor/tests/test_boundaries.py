import pytest

from trajectory_editor.boundaries import token_boundaries


@pytest.mark.parametrize("text,expected", [
    ("", set()),
    ("word", set()),
    ('”)] ', set()),
    ("\r", set()),
    ("Ċ", set()),  # Vocabulary spelling is not a decoded newline.
    (".", {"sentence"}),
    ("!", {"sentence"}),
    ("?", {"sentence"}),
    ('Dr.") Next', {"sentence"}),
    ("3.14", {"sentence"}),
    ("\n", {"newline"}),
    ("\n\n", {"newline"}),
    ('"\n\n', {"newline"}),
    ("\nNext", {"newline"}),
    (".\n\n", {"sentence", "newline"}),
])
def test_boundaries_match_anywhere_in_decoded_token(text, expected):
    assert token_boundaries(text) == expected
