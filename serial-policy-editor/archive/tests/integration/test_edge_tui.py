from __future__ import annotations


from trajectory_editor.edge_tui import _edge_header, read_live_edge_command


def test_surface_returns_the_raw_command_on_enter() -> None:
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        pipe.send_text("n 12\n")
        result = read_live_edge_command(
            episode_id="demo",
            boundary=7,
            current_budget=24,
            remaining_tokens=0,
            sampler_summary="temp=0.8 top_k=40 seed=9",
            input_device=pipe,
            output_device=DummyOutput(),
        )

    assert result == 'n 12'


def test_surface_header_keeps_checkpoint_context_visible() -> None:
    fragments = _edge_header(
        episode_id="demo",
        boundary=7,
        current_budget=24,
        remaining_tokens=3,
        sampler_summary="temp=0.8 top_k=40 seed=9",
    )
    rendered = "".join(text for _style, text in fragments)

    assert 'LIVE EDGE' in rendered
    assert 'demo' in rendered
    assert 'boundary 7' in rendered
    assert '3 tokens remaining' in rendered
    assert 'n N' in rendered
    assert 's random-seed' in rendered
    assert 'temp=0.8 top_k=40 seed=9' in rendered
