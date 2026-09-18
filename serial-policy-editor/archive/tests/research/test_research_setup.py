from __future__ import annotations

from pathlib import Path

from trajectory_editor.runtime_setup import RuntimePlan, apply_setup_command, build_controller_stack


def test_research_learning_and_preference_panels_remain_explicit():
    plan = RuntimePlan()

    assert plan.learning_dead_zone_rank == 0
    assert plan.token_preference_dead_zone_rank == 0
    assert apply_setup_command("learning", plan) == "show-learning"
    assert apply_setup_command("preference", plan) == "show-preference"

    apply_setup_command(
        "group on rate=.2 gate=sampler groups=concrete,abstract from_write=on",
        plan,
    )
    apply_setup_command(
        "preference on dimension=32 learning_scheme=fisher-kl-v2 "
        "fast_slow=on projection_seed=random",
        plan,
    )

    assert plan.online_learning is True
    assert plan.learning_rate == 0.2
    assert plan.learning_gate == "sampler"
    assert plan.learnable_groups == ("concrete", "abstract")
    assert plan.learn_from_write is True
    assert plan.token_preference is True
    assert plan.token_preference_dimension == 32
    assert plan.token_preference_learning_scheme == "fisher-kl-v2"
    assert plan.token_preference_fast_slow is True
    assert plan.token_preference_random_projection_seed is True


def test_research_controller_stack_exposes_research_control_surfaces():
    plan = RuntimePlan(
        biases=Path("biases.json"),
        reference=Path("reference.yaml"),
        activation_vector=Path("style.json"),
        online_learning=True,
        token_preference=True,
    )

    stack = build_controller_stack(plan=plan)
    names = [entry.name for entry in stack.entries if entry.phase == "policy"]
    assert names == [
        "base model",
        "history penalties",
        "output-head steering",
        "manual biases/groups",
        "reference prior",
        "token preference actuator",
        "group control",
        "sampler / token draw",
    ]
