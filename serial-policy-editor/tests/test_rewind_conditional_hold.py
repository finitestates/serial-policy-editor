import pytest

from tests.test_hold_expectations import runtime
from tests.test_replay_eog import create
from trajectory_editor.episode_actions import Hold, Write
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_projector import project_procedure


@pytest.mark.parametrize("condition", ["sentence", "newline"])
@pytest.mark.parametrize("retained", [2, 4])
def test_rewind_derives_plain_partial_hold_but_preserves_full_hold(tmp_path, condition, retained):
    source = runtime([1, 3, 5, 1])
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = create(store, source)
        store.record_action(identifier, 0, source.apply(Hold(4, condition)))
        store.record_action(identifier, 1, source.apply(Write("hello")))
        original_tokens = store.tokens(identifier)[:retained]
        source.rewind_to(retained)
        store.rewind_to(identifier, retained, visible_text=source.backend.render(source.visible_token_ids),
                        max_tokens=source.max_tokens)
        expected = Hold(2) if retained == 2 else Hold(4, condition)
        action, expectation = store.replay_tape(identifier)[0]
        assert action == expected
        assert store.replay_procedure(identifier)[0]["action"] == expected
        assert len(store.actions(identifier)) == 1
        assert store.tokens(identifier) == original_tokens
        assert expectation.token_ids == tuple(source.visible_token_ids)
        text = project_procedure(store, identifier)
        if retained == 2:
            assert "0 : h 2 #" in text
            assert "h . " not in text and "h | " not in text
            # A new boundary character in a counterfactual must not shorten
            # the edited finite hold.
            target = runtime([4 if condition == "newline" else 2, 3])
            result = target.apply(action, replay=True, expectation=expectation,
                                  divergence_policy="ballistic")
            assert len(result.visible_token_ids) == 2
            assert result.stop_reason == "requested-length"
        else:
            assert ("h . 4" if condition == "sentence" else "h | 4") in text
