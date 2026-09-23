from __future__ import annotations

from pathlib import Path

import pytest

from trajectory_editor.edge_commands import (
    BranchesCommand,
    BudgetCommand,
    ContinueCommand,
    EdgeCommandParseError,
    EndCommand,
    ExportCommand,
    ForkCommand,
    ForkMapCommand,
    ListCommand,
    NameCommand,
    NewCommand,
    ProjectCommand,
    QuitCommand,
    ReplayCommand,
    ReplaySelectionCommand,
    RewindCommand,
    SamplerCommand,
    SaveCommand,
    SaveFamilyCommand,
    SwitchCommand,
    parse_edge_command,
)

pytestmark = pytest.mark.current_workflow

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ContinueCommand()),
        ("   ", ContinueCommand()),
        ("c", ContinueCommand()),
        ("CONTINUE", ContinueCommand()),
        ("n 17", BudgetCommand(17)),
        ("next +3", BudgetCommand(3)),
        ("n off", BudgetCommand(None)),
        ("next none", BudgetCommand(None)),
        ("s", SamplerCommand()),
        ("s temperature=.8 top_k=20", SamplerCommand("temperature=.8 top_k=20")),
        ("SAMPLER random-seed", SamplerCommand("random-seed")),
        ("rewind 12", RewindCommand(12)),
        ("f 0", ForkCommand(0)),
        ("FORK -2", ForkCommand(-2)),
        ("switch branch-abc", SwitchCommand("branch-abc")),
        ("switch 7", SwitchCommand("7")),
        ("#7", SwitchCommand("#7")),
        ("new", NewCommand(None)),
        ("new   prompt with  spaces", NewCommand("prompt with  spaces")),
        ("e", EndCommand()),
        ("END", EndCommand()),
        ("q", QuitCommand()),
        ("quit", QuitCommand()),
        ("p", ProjectCommand()),
        ("review", ProjectCommand()),
        ("forkmap", ForkMapCommand()),
        ("fork-map", ForkMapCommand()),
        ("branches", BranchesCommand()),
        ("ls", ListCommand(False)),
        ("ls all", ListCommand(True)),
        ("name Episode Title", NameCommand("Episode Title")),
        ("export tape.jsonl", ExportCommand(Path("tape.jsonl"))),
        ("save workspace.sqlite #7", SaveCommand(Path("workspace.sqlite"), "#7")),
        ("save-family workspace.sqlite", SaveFamilyCommand(Path("workspace.sqlite"))),
        (
            "save family workspace.sqlite root-abc",
            SaveFamilyCommand(Path("workspace.sqlite"), "root-abc"),
        ),
        ("spr #12", ReplayCommand("#12")),
        ("spr #12 m", ReplaySelectionCommand("#12")),
        ("replay source-id m", ReplaySelectionCommand("source-id")),
        ("REPLAY episode-id --until 31", ReplayCommand("episode-id", 31)),
    ],
)
def test_parse_edge_command_extracts_typed_values(raw, expected):
    assert parse_edge_command(raw) == expected


@pytest.mark.parametrize("raw", ["rewind nope", "fork 1.5", "replay source --until later"])
def test_malformed_numeric_boundaries_are_parse_errors(raw):
    with pytest.raises(EdgeCommandParseError):
        parse_edge_command(raw)


@pytest.mark.parametrize("raw", ["n 0", "n -1", "next nope", "n 2 extra"])
def test_malformed_budgets_are_parse_errors(raw):
    with pytest.raises(EdgeCommandParseError):
        parse_edge_command(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "export",
        "export a b",
        "save",
        "save workspace a b",
        "save-family",
        "save-family workspace a b",
        "save family",
        "save family workspace a b c",
        "name",
        "spr",
        "spr source extra",
        "spr source until 3",
    ],
)
def test_malformed_save_and_replay_forms_are_parse_errors(raw):
    with pytest.raises(EdgeCommandParseError):
        parse_edge_command(raw)


def test_references_are_not_resolved_or_normalized():
    assert parse_edge_command("#004") == SwitchCommand("#004")
    assert parse_edge_command("switch #004") == SwitchCommand("#004")
    assert parse_edge_command("spr #004") == ReplayCommand("#004")
    assert parse_edge_command("save workspace #004") == SaveCommand(Path("workspace"), "#004")


@pytest.mark.parametrize(
    "raw",
    [
        "unknown",
        "continue extra",
        "switch",
        "fork-map extra",
        "branches extra",
        "ls everything",
    ],
)
def test_unknown_or_wrong_arity_is_a_domain_parse_error(raw):
    with pytest.raises(EdgeCommandParseError):
        parse_edge_command(raw)
