"""Shared EDGE command labels for the live and plain presentations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeHelpItem:
    command: str
    description: str


_COMMON_START = (
    EdgeHelpItem("c / continue", "resume the current tranche"),
    EdgeHelpItem("n N / n off", "set an allowance or remove the budget"),
    EdgeHelpItem("s key=value", "change sampler settings"),
    EdgeHelpItem("f N", "fork at boundary N"),
)

_DURABLE = (
    EdgeHelpItem("ls / ls all", "list open / all episodes"),
    EdgeHelpItem("#N", "switch episode"),
    EdgeHelpItem("new TEXT", "start an unrelated episode"),
    EdgeHelpItem("name TITLE", "rename this episode"),
    EdgeHelpItem("rewind N", "delete continuation from token N"),
    *_COMMON_START[:3],
    EdgeHelpItem("s random-seed", "choose and record a new random seed"),
    *_COMMON_START[3:],
    EdgeHelpItem("fm", "show the fork map and choose a boundary"),
    EdgeHelpItem("spr ID [--until N | m]", "replay from another episode"),
    EdgeHelpItem("p / project", "view the episode record"),
    EdgeHelpItem("e / end", "end and seal the episode"),
    EdgeHelpItem("q / quit", "leave without sealing"),
)

_EPHEMERAL = (
    EdgeHelpItem("branches", "show retained live branches"),
    EdgeHelpItem("#N", "switch to any retained root or branch"),
    EdgeHelpItem("switch N", "compatible branch-switch alias"),
    EdgeHelpItem("new TEXT", "start an unrelated prompt root"),
    EdgeHelpItem("rewind N", "trim this branch back to token N"),
    *_COMMON_START,
    EdgeHelpItem("fm", "show the fork map and choose a boundary"),
    EdgeHelpItem("export FILE", "write the selected portable tape"),
    EdgeHelpItem("save WORKSPACE [ID]", "materialize this branch"),
    EdgeHelpItem("save-family WORKSPACE [ROOT_ID]", "materialize this live family"),
    EdgeHelpItem("e / end", "end this branch and show its text"),
    EdgeHelpItem("q / quit", "discard the whole live session"),
)


def edge_help(mode: str) -> tuple[EdgeHelpItem, ...]:
    return _EPHEMERAL if mode == "session" else _DURABLE
