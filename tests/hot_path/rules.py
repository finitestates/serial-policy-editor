"""Hot-path rules. OWNER-EDITED ONLY (see CODEOWNERS).

Agents: you may not edit this file. If a rule blocks correct work, stop and
explain in the PR which rule, which function, and why. Adding an allowlist
entry is the owner's decision, not yours.
"""

# --------------------------------------------------------------------------
# Where the interactive path starts. Everything reachable from these is
# "interactive": keystrokes, Enter, [ / ], chord advance/rewind/select,
# review navigation, warm selection.
# --------------------------------------------------------------------------
INTERACTIVE_ROOTS = [
    "chord:chord_menu",
    "chord:Chord.__init__",                      # opening a chord
    "chord:Chord.advance",
    "chord:Chord.rewind",
    "chord:Chord.promote",
    "chord:Chord.select",
    "chord:Chord.discard",
    "chord:Chord.display",
    "episode_engine:EpisodeEngine.apply",
    "episode_engine:EpisodeEngine.observe",
    "episode_engine:EpisodeEngine.speculate_accept",
    "episode_engine:EpisodeEngine.discard_speculative_accept",
    "episode_engine:EpisodeEngine.has_prepared_accept",
    "episode_engine:EpisodeEngine.rewind_to",
    "episode_ui:InteractivePolicy.choose",
    "episode_ui:_ContextRenderCursor.update",
    "episode_ui:_ContextRenderCursor.prewarm",
    "persistent_tui:PersistentTerminalSession._read",
    "persistent_tui:PersistentTerminalSession._after_key_press",   # every keystroke
    "persistent_tui:PersistentTerminalSession._before_render",     # every frame
    "persistent_tui:PersistentTerminalSession._rendered",          # every frame
    "persistent_tui:PersistentTerminalSession._refresh_warm_target",
    "persistent_tui:PersistentTerminalSession._queue_warm_selection",
    "persistent_tui:PersistentTerminalSession._submit",
    "persistent_tui:PersistentTerminalSession._show",
    "persistent_tui:PersistentTerminalSession._preview",
    "live_tui:LiveChoiceView.update",
]

# Functions that only record that something changed. They may reposition a
# cache when that is their job, but they must not READ model output.
NO_EAGER_READS = [
    "chord:Chord.rewind",
    "chord:Chord._activate",
    "chord:Chord.discard",
    "episode_engine:EpisodeEngine.rewind_to",
    "episode_engine:EpisodeEngine._commit_token",
    "episode_engine:EpisodeEngine._ensure_backend_positioned",
    "episode_engine:EpisodeEngine.adopt_preview_state",
    "episode_ui:_ContextRenderCursor.prewarm",
]

# Functions that must do no backend work at all (rollback/commit of a
# speculation are free by contract and are therefore permitted).
PURE_BOOKKEEPING = [
    "episode_engine:EpisodeEngine._invalidate_observation",
    "episode_engine:EpisodeEngine._invalidate_guidance",
    "episode_engine:EpisodeEngine.discard_speculative_accept",
    "episode_engine:EpisodeEngine.terminate",
]

# --------------------------------------------------------------------------
# Symbols. (receiver_kind, name). receiver_kind "any" matches every receiver.
# --------------------------------------------------------------------------
NEVER_ON_INTERACTIVE_PATH = {
    ("any", "save_state"),        # llama-cpp-python full context copy
    ("any", "load_state"),
    ("any", "snapshot_state"),    # SPE wrappers around the above / KV deepcopy
    ("any", "restore_state"),
    ("any", "copy.deepcopy"),     # KV caches, tensors, anything large
    ("any", "hashlib"),           # hashing token prefixes for in-process keys
    ("any", "threading.Thread"),  # background replays competing for the GIL
    ("any", "threading.Timer"),
}

FULL_PREFILL = {
    ("backend", "reset"),         # re-evaluates the entire prefix
}

EAGER_READS = {
    ("any", "observe"),
    ("any", "last_logits"),
    ("engine", "candidates"),
    ("engine", "policy_candidates"),
    ("backend", "render"),
    ("backend", "token_text"),
    ("backend", "new_text_stream"),
    ("any", "_append_stream"),
    ("any", "_rebuild_stream_only"),
}

ANY_BACKEND_WORK = EAGER_READS | {
    ("backend", name)
    for name in (
        "eval", "reset", "branch_to_prefix", "truncate_to", "speculate",
        "snapshot_state", "restore_state", "open_slots", "use_slot",
        "truncate_slot", "close_slots",
    )
}

# --------------------------------------------------------------------------
# How receivers are classified. Extend only with owner review.
# --------------------------------------------------------------------------
RECEIVER_KINDS = {
    "backend": {"backend", "guidance_backend"},
    "engine": {"engine", "original", "preview", "_engine"},
}
CLASS_KINDS = {  # what `self` means inside these classes
    "backend": {"decoder:LlamaCppDecoder", "transformers_backend:TransformersBackend"},
    "engine": {"episode_engine:EpisodeEngine"},
}
RECEIVER_TYPES = {  # attribute name -> classes it may hold (for graph edges)
    "engine": ["episode_engine:EpisodeEngine"],
    "original": ["episode_engine:EpisodeEngine"],
    "preview": ["episode_engine:EpisodeEngine"],
    "_engine": ["episode_engine:EpisodeEngine"],
    "backend": ["decoder:LlamaCppDecoder", "transformers_backend:TransformersBackend"],
    "guidance_backend": ["decoder:LlamaCppDecoder", "transformers_backend:TransformersBackend"],
    "_context_cursor": ["episode_ui:_ContextRenderCursor"],
    "chord": ["chord:Chord"],
    "choice_view": ["live_tui:LiveChoiceView"],
    "io": ["persistent_tui:PersistentTerminalSession"],
}

# --------------------------------------------------------------------------
# ALLOW: accepted permanently. (rule, function, symbol) -> why + bounding test.
# DEBT:  tolerated for now, tagged with the task that must remove it. DEBT can
#        only shrink: a DEBT entry that no longer matches FAILS the suite (so
#        fixed debt must be deleted), and nothing may be added to it except by
#        the owner. Neither list may contain the same key.
# --------------------------------------------------------------------------
_FALLBACK_TEST = "tests/hot_path/test_runtime_budgets.py::test_chord_scenarios_never_full_prefill"
ALLOW = {
    ("no-full-prefill", "decoder:LlamaCppDecoder.branch_to_prefix", "reset"): {
        "why": "Backend-internal fallback when the target is not a cached prefix. Runtime budgets "
               "require zero full prefills in every interactive scenario.",
        "test": _FALLBACK_TEST,
    },
    ("no-full-prefill", "transformers_backend:TransformersBackend.branch_to_prefix", "reset"): {
        "why": "Same backend-internal fallback as llama.cpp; bounded by the same runtime test.",
        "test": _FALLBACK_TEST,
    },
    ("no-full-prefill", "chord:_position", "reset"): {
        "why": "Only reached for backends without branch_to_prefix; real backends take the branch path.",
        "test": _FALLBACK_TEST,
    },
    ("no-full-prefill", "episode_engine:EpisodeEngine.rewind_to", "reset"): {
        "why": "Only reached for backends without branch_to_prefix.",
        "test": _FALLBACK_TEST,
    },
    ("no-full-prefill", "episode_engine:EpisodeEngine._ensure_backend_positioned", "reset"): {
        "why": "Only reached for backends without branch_to_prefix.",
        "test": _FALLBACK_TEST,
    },
    ("no-full-prefill", "episode_engine:EpisodeEngine.__init__", "reset"): {
        "why": "Construction without backend_positioned; chord previews pass backend_positioned=True.",
        "test": _FALLBACK_TEST,
    },
    ("no-full-prefill", "episode_engine:EpisodeEngine._position_guidance", "reset"): {
        "why": "CFG guidance prompt changed; the unconditional context genuinely differs.",
        "test": _FALLBACK_TEST,
    },
    ("no-full-prefill", "episode_engine:EpisodeEngine._prepare_activation_runtime", "reset"): {
        "why": "Control vector changed; every cached activation is invalid.",
        "test": _FALLBACK_TEST,
    },
    ("never-on-interactive-path", "core.sampling:position_uniform", "hashlib"): {
        "why": "Seeded deterministic draw over a fixed-size key, not a token prefix.",
        "test": "tests/hot_path/test_runtime_budgets.py::test_observe_budget",
    },
    ("never-on-interactive-path", "core.sampling:position_uniform_token", "hashlib"): {
        "why": "Seeded deterministic draw over a fixed-size key, not a token prefix.",
        "test": "tests/hot_path/test_runtime_budgets.py::test_observe_budget",
    },
    ("never-on-interactive-path", "bias_commands:_name", "hashlib"): {
        "why": "Runs once per typed bias command on the command text, not per token.",
        "test": "tests/hot_path/test_runtime_budgets.py::test_observe_budget",
    },
    ("never-on-interactive-path", "bias_commands:resolve_target", "hashlib"): {
        "why": "Runs once per typed bias command on the command text, not per token.",
        "test": "tests/hot_path/test_runtime_budgets.py::test_observe_budget",
    },
    ("no-swallowed-backend-errors", "episode_ui:InteractivePolicy._search", "tokenize"): {
        "why": "Tokenizing user-typed search text; a failure is a user input error, not a hidden cost.",
        "test": "tests/hot_path/test_runtime_budgets.py::test_observe_budget",
    },
    ("invalidate-dont-refresh", "episode_engine:EpisodeEngine._evidence", "token_text"): {
        "why": "One-token text for the committed token's outcome record; not a model read.",
        "test": "tests/hot_path/test_runtime_budgets.py::test_speculation_hit_and_miss_budgets",
    },
}

ALLOW[("invalidate-dont-refresh", "core.backend:require_inference_backend", "last_logits")] = {
    "why": "callable(backend.last_logits) is a capability check at construction; nothing is read.",
    "test": "tests/hot_path/test_runtime_budgets.py::test_observe_budget",
}

DEBT = {
    # ---- Task A: chord ------------------------------------------------------
    # ---- Task B: speculation ------------------------------------------------
    # ---- Task C: review cursor ----------------------------------------------
    # ---- U: found by the checker, not yet triaged by the owner ---------------
    # Chord open builds k preview engines; each hashes the whole initial prompt.
    ("never-on-interactive-path", "episode_hash:token_prefix_sha256", "hashlib"): "U",
    # Assigning `sampling` (chord open, every sampler edit) hashes the steering vector.
    ("never-on-interactive-path", "activation_vectors:steering_vector_digest_for", "hashlib"): "U",
    # Output-head steering hashes the vector on every observe() to look up a cache.
    ("never-on-interactive-path", "decoder:LlamaCppDecoder.activation_logit_adjustments", "hashlib"): "U",
    ("never-on-interactive-path", "transformers_backend:TransformersBackend.activation_logit_adjustments", "hashlib"): "U",
    ("never-on-interactive-path", "transformers_backend:TransformersBackend.set_hidden_state_vector", "hashlib"): "U",
    # First observe() with a model-bound vector hashes the entire model FILE (cached after).
    ("never-on-interactive-path", "model_hash:sha256_path", "hashlib"): "U",
    # adopt_preview_state constructs an engine; __init__ renders when initial_text is not a str.
    ("invalidate-dont-refresh", "episode_engine:EpisodeEngine.__init__", "render"): "U",
}

# --------------------------------------------------------------------------
# Every method of these classes must be interactive (reachable from a root)
# or explicitly COLD. New methods fail until classified by the owner.
# --------------------------------------------------------------------------
WATCHED_CLASSES = [
    "chord:Chord",
    "episode_engine:EpisodeEngine",
    "episode_ui:_ContextRenderCursor",
    "persistent_tui:PersistentTerminalSession",
    "decoder:LlamaCppDecoder",
    "transformers_backend:TransformersBackend",
]
COLD = {  # baseline classification; the owner adds new entries
    "decoder:LlamaCppDecoder.__init__",
    "decoder:LlamaCppDecoder.close",
    "decoder:LlamaCppDecoder.model_id",
    "episode_engine:EpisodeEngine.resume",
    "episode_engine:EpisodeEngine.terminal_reason",
    "episode_engine:EpisodeEngine.terminate",
    "episode_engine:EpisodeEngine.text",
    "episode_ui:_ContextRenderCursor.__init__",
    "persistent_tui:PersistentTerminalSession.__enter__",
    "persistent_tui:PersistentTerminalSession.__exit__",
    "persistent_tui:PersistentTerminalSession.__init__",
    "persistent_tui:PersistentTerminalSession._run",
    "persistent_tui:PersistentTerminalSession._run_application",
    "persistent_tui:PersistentTerminalSession._ui_exception",
    "persistent_tui:PersistentTerminalSession._write",
    "persistent_tui:PersistentTerminalSession.read",
    "persistent_tui:PersistentTerminalSession.read_edge",
    "persistent_tui:PersistentTerminalSession.read_multiline_prompt",
    "persistent_tui:PersistentTerminalSession.write",
    "transformers_backend:TransformersBackend.__init__",
    "transformers_backend:TransformersBackend._apply_execution_controls",
    "transformers_backend:TransformersBackend._build_quantization_config",
    "transformers_backend:TransformersBackend._discover_eog_ids",
    "transformers_backend:TransformersBackend._infer_input_device",
    "transformers_backend:TransformersBackend._resolve_device",
    "transformers_backend:TransformersBackend._run_hidden_state_capture",
    "transformers_backend:TransformersBackend.activation_control_vector_layer_count",
    "transformers_backend:TransformersBackend.activation_control_vector_width",
    "transformers_backend:TransformersBackend.activation_snapshot",
    "transformers_backend:TransformersBackend.hidden_state_capabilities",
    "transformers_backend:TransformersBackend.hidden_state_runtime_layer_range",
    "transformers_backend:TransformersBackend.hidden_state_snapshot",
    "transformers_backend:TransformersBackend.model_id",
}
