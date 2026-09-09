"""Plain projection uses authoritative saved text without loading evidence."""
from unittest.mock import Mock

import pytest

from trajectory_editor.domain import EditorError
from trajectory_editor.episode_projector import project_episode


@pytest.mark.parametrize('include_initial', [False, True])
@pytest.mark.parametrize('with_lineage', [False, True])
@pytest.mark.parametrize('visible_text', ['', ' 世界\n saved text'])
def test_plain_projection_skips_evidence(include_initial, with_lineage, visible_text):
    from unittest.mock import patch
    store = Mock()
    store.get_episode.return_value = {
        'initial_text': 'Prompt\n', 'visible_text': visible_text,
        'status': 'completed', 'terminal_reason': 'model-eog',
    }
    for method in (store.tokens, store.actions, store.interactions):
        method.side_effect = AssertionError('plain projection must not load evidence')
    with patch('trajectory_editor.episode_projector.project_lineage', return_value='lineage') as lineage:
        result = project_episode(store, 'episode', include_initial=include_initial,
                                 with_lineage=with_lineage)
        assert lineage.call_count == int(with_lineage)
    assert result.text == ('Prompt\n' if include_initial else '') + visible_text + ('\n\nlineage' if with_lineage else '')
    assert result.episode_id == 'episode'
    assert result.annotations == ()
    assert result.status == 'completed'
    assert result.terminal_reason == 'model-eog'


@pytest.mark.parametrize('options', [
    {'annotations': 'inline'}, {'annotations': 'footnotes'},
    {'with_loss': True}, {'with_rank': True}, {'with_policy_rank': True},
    {'full_evidence': True}, {'with_model_probs': True},
])
def test_evidence_options_keep_detailed_projection(options):
    store = Mock()
    store.get_episode.return_value = {
        'initial_text': 'P', 'visible_text': 'saved', 'status': 'open',
    }
    store.tokens.return_value = []
    store.actions.return_value = []
    store.interactions.return_value = []
    result = project_episode(store, 'episode', **options)
    assert result.text == 'P'
    for method in (store.tokens, store.actions, store.interactions):
        method.assert_called_once_with('episode')


def test_invalid_annotation_mode_still_rejected_before_loading():
    store = Mock()
    with pytest.raises(EditorError, match='annotations must be'):
        project_episode(store, 'episode', annotations='invalid')
    store.get_episode.assert_not_called()
