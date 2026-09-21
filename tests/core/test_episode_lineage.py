from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trajectory_editor.episode_lineage import EpisodeRelation, build_lineage


def relation(
    episode_id: str,
    *,
    parent_id: str | None = None,
    mode: str = "interactive",
    source_id: str | None = None,
    created: int = 0,
    boundary: int | None = None,
) -> EpisodeRelation:
    return EpisodeRelation(
        episode_id=episode_id,
        parent_id=parent_id,
        fork_boundary=boundary,
        mode=mode,
        spr_source_id=source_id,
        status="complete",
        creation_key=created,
        terminal_reason=None,
        visible_token_count=created,
    )


def tree_ids(node):
    if node is None:
        return []
    return [node.episode_id, [tree_ids(child) for child in node.children]]


def test_ordinary_nested_fork_family_is_a_typed_tree():
    records = [
        relation("grandchild", parent_id="child", created=3, boundary=4),
        relation("root", created=1),
        relation("child", parent_id="root", created=2, boundary=2),
        relation("sibling", parent_id="root", created=4, boundary=1),
    ]

    view = build_lineage(records, "grandchild")

    assert view.selected_record.episode_id == "grandchild"
    assert view.ordinary_family_root.episode_id == "root"
    assert tree_ids(view.ordinary_fork_tree) == [
        "root",
        [["child", [["grandchild", []]]], ["sibling", []]],
    ]
    assert all(not node.record.is_replay for node in [view.ordinary_fork_tree])


def test_replay_is_related_provenance_but_not_an_ordinary_tree_node():
    records = [
        relation("root", created=1),
        relation("branch", parent_id="root", created=2),
        relation(
            "replay",
            parent_id="root",
            source_id="root",
            created=3,
        ),
    ]

    view = build_lineage(records, "branch")

    assert [record.episode_id for record in view.related_replays] == ["replay"]
    assert tree_ids(view.ordinary_fork_tree) == ["root", [["branch", []]]]
    assert "replay" not in {
        view.ordinary_fork_tree.episode_id,
        *(child.episode_id for child in view.ordinary_fork_tree.children),
    }


def test_ordinary_forks_directly_derived_from_related_replays_are_separate():
    records = [
        relation("root", created=1),
        relation("replay", parent_id="root", mode="serial-policy-replay", created=2),
        relation("from-replay", parent_id="replay", created=4, boundary=5),
        relation("ordinary-child", parent_id="root", created=3),
        relation("nested-from-fork", parent_id="from-replay", created=5),
    ]

    view = build_lineage(records, "root")

    assert [record.episode_id for record in view.replay_derived_forks] == [
        "from-replay"
    ]
    assert tree_ids(view.ordinary_fork_tree) == [
        "root",
        [["ordinary-child", []]],
    ]


def test_selecting_a_replay_locates_its_ordinary_family_seed():
    records = [
        relation("root", created=1),
        relation("child", parent_id="root", created=2),
        relation(
            "replay",
            parent_id="child",
            mode="serial-policy-replay",
            source_id="root",
            created=3,
        ),
        relation("from-replay", parent_id="replay", created=4),
    ]

    view = build_lineage(records, "replay")

    assert view.ordinary_family_root_id == "root"
    assert tree_ids(view.ordinary_fork_tree) == ["root", [["child", []]]]
    assert [record.episode_id for record in view.related_replays] == ["replay"]
    assert [record.episode_id for record in view.replay_derived_forks] == ["from-replay"]


def test_siblings_and_related_records_are_sorted_by_creation_then_id():
    records = [
        relation("z-child", parent_id="root", created=2),
        relation("root", created=1),
        relation("a-child", parent_id="root", created=2),
        relation("replay-z", parent_id="root", mode="serial-policy-replay", created=4),
        relation("replay-a", parent_id="root", mode="serial-policy-replay", created=4),
        relation(
            "replay-early",
            parent_id="root",
            mode="serial-policy-replay",
            created=3,
        ),
    ]

    view = build_lineage(reversed(records), "root")

    assert [child.episode_id for child in view.ordinary_fork_tree.children] == ["a-child", "z-child"]
    assert [record.episode_id for record in view.related_replays] == [
        "replay-early",
        "replay-a",
        "replay-z",
    ]


def test_missing_references_produce_partial_inspectable_results():
    records = [
        relation("orphan", parent_id="missing-parent", created=1),
        relation(
            "source-missing",
            mode="serial-policy-replay",
            source_id="missing-source",
            created=2,
        ),
    ]

    orphan = build_lineage(records, "orphan")
    replay = build_lineage(records, "source-missing")

    assert orphan.ordinary_family_root_id == "orphan"
    assert orphan.ordinary_fork_tree.episode_id == "orphan"
    assert replay.ordinary_family_root is None
    assert replay.ordinary_fork_tree is None
    assert [record.episode_id for record in replay.related_replays] == ["source-missing"]


def test_parent_cycles_are_finite_and_do_not_crash():
    records = [
        relation("a", parent_id="b", created=2),
        relation("b", parent_id="a", created=1),
        relation("child", parent_id="a", created=3),
    ]

    view = build_lineage(records, "child")

    assert view.ordinary_family_root_id == "b"
    assert tree_ids(view.ordinary_fork_tree) == ["b", [["a", [["child", []]]]]]


def test_records_and_views_are_immutable():
    record = relation("root")
    view = build_lineage([record], "root")

    with pytest.raises(FrozenInstanceError):
        record.status = "running"
    with pytest.raises(FrozenInstanceError):
        view.related_replays = ()


@pytest.mark.parametrize(
    "invalid",
    [
        {"episode_id": ""},
        {"episode_id": "contains whitespace"},
        {"episode_id": []},
        {"parent_id": []},
        {"spr_source_id": {}},
        {"fork_boundary": -1},
        {"fork_boundary": True},
        {"mode": None},
        {"mode": ""},
        {"status": 1},
        {"status": " "},
        {"visible_token_count": -1},
        {"visible_token_count": False},
    ],
)
def test_relation_constructor_rejects_malformed_typed_values(invalid):
    with pytest.raises((TypeError, ValueError)):
        EpisodeRelation(episode_id="valid", **invalid)


def test_duplicate_relation_ids_are_rejected_instead_of_selected():
    with pytest.raises(ValueError, match="duplicate episode relation ID"):
        build_lineage(
            [relation("same", created=1), relation("same", created=2)],
            "same",
        )
