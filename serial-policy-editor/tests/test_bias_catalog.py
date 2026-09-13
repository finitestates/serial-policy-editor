import json
from pathlib import Path
from unittest.mock import patch

import pytest

from trajectory_editor.bias_catalog import (
    ALLOCATIONS,
    BiasCatalog,
    CompileOptions,
    allocate_route,
    build_prefix_reference_stats,
    compile_catalog,
    compile_term,
    generate_forms,
    load_yaml_source,
)
from trajectory_editor.bias_cli import main as bias_main
from trajectory_editor.domain import EditorError


class CatalogBackend:
    pieces = {
        0: "<EOG>",
        1: "shadow",
        2: " shadow",
        3: "Shadow",
        4: " shadows",
        5: "shadows",
        6: "shadowing",
        7: " shadowing",
        8: "ing",
        9: "sh",
        10: "adow",
        11: " hear",
        12: "th",
        13: "hearth",
        14: "anchor",
        15: " steamship",
        16: "steamship",
        17: "port",
        18: " of",
        19: " call",
        20: " port",
        21: "port of call",
        22: " Shadow",
        23: "Shadows",
        24: " Shadows",
    }

    canonical = {
        "shadow": (1,),
        " shadow": (2,),
        "Shadow": (3,),
        " Shadow": (22,),
        "shadows": (5,),
        " shadows": (4,),
        "Shadows": (23,),
        " Shadows": (24,),
        "shadowing": (6,),
        " shadowing": (7,),
        "hearth": (13,),
        " hearth": (11, 12),
        "anchor": (14,),
        "steamship": (16,),
        " steamship": (15,),
        "port of call": (17, 18, 19),
        " port of call": (20, 18, 19),
    }

    def vocabulary_size(self):
        return len(self.pieces)

    def token_text(self, token_id):
        return self.pieces[int(token_id)]

    def is_eog(self, token_id):
        return int(token_id) == 0

    def tokenize(self, text, *, add_bos=False, special=False):
        assert not add_bos and not special
        return list(self.canonical.get(text, ()))

    def render(self, token_ids, *, special=False):
        assert not special
        return "".join(self.pieces[int(token)] for token in token_ids)

    def provenance(self, *, include_model_sha256=False):
        del include_model_sha256
        return {"backend": "catalog-test", "vocabulary_size": self.vocabulary_size()}


@pytest.fixture
def backend():
    return CatalogBackend()


class RouteRankingBackend:
    pieces = {
        0: "<EOG>",
        1: "an",
        2: "other",
        3: "another",
        4: "aj",
        5: "ar",
    }
    canonical = {
        "another": (1, 2),
        "ajar": (4, 5),
    }

    def vocabulary_size(self):
        return len(self.pieces)

    def token_text(self, token_id):
        return self.pieces[int(token_id)]

    def is_eog(self, token_id):
        return int(token_id) == 0

    def tokenize(self, text, *, add_bos=False, special=False):
        assert not add_bos and not special
        return list(self.canonical.get(text, ()))

    def render(self, token_ids, *, special=False):
        assert not special
        return "".join(self.pieces[int(token)] for token in token_ids)

    def provenance(self, *, include_model_sha256=False):
        del include_model_sha256
        return {"backend": "route-ranking-test", "vocabulary_size": self.vocabulary_size()}


def test_yaml_scalar_quotes_and_implicit_global_group(tmp_path, backend):
    path = tmp_path / "terms.yaml"
    path.write_text(
        "terms:\n"
        "  - shadow\n"
        "  - \"shadow\"\n"
        "  - 'shadow'\n"
        "defaults:\n"
        "  cases: [original]\n"
        "  plural: false\n",
        encoding="utf-8",
    )
    source = load_yaml_source(path)
    catalog = compile_catalog(source, backend)

    assert set(catalog.entries) == {"shadow", "global"}
    assert catalog.require("global").members == ("shadow",)
    route_texts = {text for route in catalog.require("shadow").routes for text in route.texts}
    assert route_texts == {"shadow", " shadow"}


def test_yaml_lexical_scalars_do_not_change_with_yaml_quote_style(tmp_path):
    path = tmp_path / "scalar-forms.yaml"
    path.write_text(
        "terms:\n"
        "  - yes\n"
        "  - \"yes\"\n"
        "  - 'yes'\n",
        encoding="utf-8",
    )
    assert load_yaml_source(path) == {"terms": ["yes", "yes", "yes"]}


def test_sentence_case_generates_natural_multiword_capitalization():
    options = CompileOptions(
        cases=("sentence",),
        leading_space=False,
        plural=False,
    )

    assert generate_forms("my favorite chair", options) == ("My favorite chair",)
    assert "My favorite chair" in generate_forms(
        "my favorite chair",
        CompileOptions(leading_space=False, plural=False),
    )


def test_title_case_keeps_possessive_suffix_lowercase():
    assert generate_forms(
        "my friend's couch",
        CompileOptions(cases=("title",), leading_space=False, plural=False),
    ) == ("My Friend's Couch",)
    assert generate_forms(
        "my friend's couch",
        CompileOptions(cases=("upper",), leading_space=False, plural=False),
    ) == ("MY FRIEND'S COUCH",)


def test_auto_beheads_short_alternate_route_heads(backend):
    entry = compile_term(
        "shadow",
        "shadow",
        backend,
        options=CompileOptions(
            cases=("original",), leading_space=False, plural=False
        ),
    )

    modes = {route.token_ids: route.mode for route in entry.routes}
    assert modes[(1,)] == "path"
    assert modes[(9, 10)] == "beheaded"

    explicit_path = compile_term(
        "shadow",
        "shadow",
        backend,
        options=CompileOptions(
            mode="path", cases=("original",), leading_space=False, plural=False
        ),
    )
    assert {route.mode for route in explicit_path.routes} == {"path"}


def test_auto_beheads_whitespace_only_route_heads():
    class SpaceHeadBackend(CatalogBackend):
        pieces = {**CatalogBackend.pieces, 25: " ", 26: "window"}
        canonical = {**CatalogBackend.canonical, "window": (26,), " window": (25, 26)}

    entry = compile_term(
        "window",
        "window",
        SpaceHeadBackend(),
        options=CompileOptions(
            cases=("original",), leading_space=True, plural=False
        ),
    )

    modes = {route.token_ids: route.mode for route in entry.routes}
    assert modes[(25, 26)] == "beheaded"


def test_cohesive_route_policy_filters_tiny_subword_alternates(backend):
    all_routes = compile_term(
        "shadow",
        "shadow",
        backend,
        options=CompileOptions(
            level="exhaustive",
            cases=("original",),
            leading_space=False,
            plural=False,
            route_policy="all",
        ),
    )
    cohesive = compile_term(
        "shadow",
        "shadow",
        backend,
        options=CompileOptions(
            level="exhaustive",
            cases=("original",),
            leading_space=False,
            plural=False,
            route_policy="cohesive",
        ),
    )

    assert (1,) in {route.token_ids for route in all_routes.routes}
    assert (9, 10) in {route.token_ids for route in all_routes.routes}
    assert (1,) in {route.token_ids for route in cohesive.routes}
    assert (9, 10) not in {route.token_ids for route in cohesive.routes}


def test_cohesive_route_policy_keeps_whitespace_separated_phrase_pieces(backend):
    entry = compile_term(
        "port of call",
        "port of call",
        backend,
        options=CompileOptions(
            level="exhaustive",
            cases=("original",),
            leading_space=False,
            plural=False,
            route_policy="cohesive",
        ),
    )

    routes = {route.token_ids for route in entry.routes}
    assert (17, 18, 19) in routes
    assert (21,) in routes
    assert {route.route_class for route in entry.routes if route.token_ids == (17, 18, 19)} == {"word_aligned"}
    assert {route.route_class for route in entry.routes if route.token_ids == (21,)} == {"direct"}


def test_tokenizer_default_route_is_not_automatically_preferred():
    entry = compile_term(
        "another",
        "another",
        RouteRankingBackend(),
        options=CompileOptions(
            level="exhaustive",
            cases=("original",),
            leading_space=False,
            plural=False,
            max_routes=2,
        ),
    )

    assert [route.token_ids for route in entry.routes] == [(3,), (1, 2)]
    assert entry.routes[0].route_class == "direct"
    assert entry.routes[0].strategies == ("preferred",)
    assert entry.routes[1].route_class == "fragmented"
    assert entry.routes[1].strategies == ("derived",)


def test_cohesive_policy_omits_fragmented_default_route_when_clean_route_exists():
    entry = compile_term(
        "another",
        "another",
        RouteRankingBackend(),
        options=CompileOptions(
            level="exhaustive",
            cases=("original",),
            leading_space=False,
            plural=False,
            route_policy="cohesive",
        ),
    )

    assert [route.token_ids for route in entry.routes] == [(3,)]
    assert not entry.warnings


def test_cohesive_policy_uses_tail_only_fallback_when_no_clean_route_exists():
    entry = compile_term(
        "ajar",
        "ajar",
        RouteRankingBackend(),
        options=CompileOptions(
            level="exhaustive",
            cases=("original",),
            leading_space=False,
            plural=False,
            route_policy="cohesive",
        ),
    )

    assert [route.token_ids for route in entry.routes] == [(4, 5)]
    assert entry.mode == "tail"
    assert entry.routes[0].mode == "tail"
    assert entry.routes[0].strategies == ("fallback",)
    assert entry.warnings == (
        "term 'ajar' has no cohesive routes; using a tail-only fallback",
    )


def test_named_group_includes_implicit_terms_and_preserves_word_phrase_modes(backend):
    catalog = compile_catalog(
        {
                "defaults": {"cases": ["original"], "plural": False, "leading_space": False},
            "groups": {"nautical": ["anchor", "steamship", "port of call"]},
        },
        backend,
    )

    group = catalog.require("nautical")
    assert group.kind == "group"
    assert group.members == ("anchor", "steamship", "port of call")
    assert catalog.require("anchor").mode == "path"
    assert catalog.require("port of call").mode == "tail"
    assert {route.mode for route in group.routes} == {"path", "tail"}


def test_standard_and_exhaustive_levels_find_bounded_alternate_routes(backend):
    standard = compile_term(
        "shadowing",
        "shadowing",
        backend,
        options=CompileOptions(
            level="standard", cases=("original",), leading_space=False,
            plural=False,
        ),
    )
    exhaustive = compile_term(
        "shadowing",
        "shadowing",
        backend,
        options=CompileOptions(
            level="exhaustive", cases=("original",), leading_space=False,
            plural=False, max_route_tokens=3,
        ),
    )

    standard_routes = {route.token_ids for route in standard.routes}
    exhaustive_routes = {route.token_ids for route in exhaustive.routes}
    assert (6,) in standard_routes
    assert (1, 8) in standard_routes
    assert (9, 10, 8) not in standard_routes
    assert (9, 10, 8) in exhaustive_routes


def test_route_budget_round_robins_alternates_across_forms():
    class Backend:
        pieces = {
            0: "<EOG>",
            1: "ab",
            2: "a",
            3: "b",
            4: "a",
            5: "b",
            6: "AB",
            7: "A",
            8: "B",
            9: "A",
            10: "B",
        }
        canonical = {"ab": (1,), "AB": (6,)}

        def vocabulary_size(self):
            return len(self.pieces)

        def token_text(self, token_id):
            return self.pieces[int(token_id)]

        def is_eog(self, token_id):
            return int(token_id) == 0

        def tokenize(self, text, *, add_bos=False, special=False):
            assert not add_bos and not special
            return list(self.canonical[text])

        def render(self, token_ids, *, special=False):
            assert not special
            return "".join(self.pieces[int(token)] for token in token_ids)

        def provenance(self, *, include_model_sha256=False):
            del include_model_sha256
            return {"backend": "round-robin-test", "vocabulary_size": self.vocabulary_size()}

    entry = compile_term(
        "ab",
        "ab",
        Backend(),
        options=CompileOptions(
            level="exhaustive", cases=("original", "upper"),
            leading_space=False, plural=False, max_routes=4,
        ),
    )

    assert [route.token_ids for route in entry.routes] == [
        (1,), (6,), (2, 3), (7, 8),
    ]


def test_max_routes_applies_across_all_generated_forms(backend):
    entry = compile_term(
        "Shadow",
        "Shadow",
        backend,
        options=CompileOptions(
            level="minimal", cases=("original", "lower"),
            leading_space=False, plural=False, max_routes=1,
        ),
    )
    assert len(entry.routes) == 1


def test_explicit_forms_and_per_term_levels(backend):
    catalog = compile_catalog(
        {
            "defaults": {"cases": ["original"], "plural": False, "level": "minimal"},
            "terms": {
                "shadowing": {"forms": ["shadowing"], "level": "exhaustive",
                              "max_route_tokens": 3},
            },
        },
        backend,
    )
    entry = catalog.require("shadowing")
    assert entry.level == "exhaustive"
    assert (9, 10, 8) in {route.token_ids for route in entry.routes}


def test_per_term_mode_and_edge_scales_override_auto(backend):
    catalog = compile_catalog(
        {
            "defaults": {
                "cases": ["original"],
                "plural": False,
                "leading_space": False,
            },
            "terms": {
                "port of call": {
                    "mode": "path",
                    "head_scale": 0.25,
                    "continuation_scale": 0.75,
                },
            },
        },
        backend,
    )

    entry = catalog.require("port of call")
    assert entry.mode == "path"
    assert {(route.mode, route.head_scale, route.continuation_scale)
            for route in entry.routes} == {("path", 0.25, 0.75)}
    restored = BiasCatalog.from_json(catalog.to_json())
    restored_route = restored.require("port of call").routes[0]
    assert restored_route.head_scale == 0.25
    assert restored_route.continuation_scale == 0.75


def test_information_allocation_is_precomputed_and_route_conserving(backend):
    entry = compile_term(
        "shadowing",
        "shadowing",
        backend,
        options=CompileOptions(
            level="exhaustive",
            cases=("original",),
            leading_space=False,
            plural=False,
            allocation="information",
            max_route_tokens=3,
        ),
    )

    assert "information" in ALLOCATIONS
    assert entry.routes
    for route in entry.routes:
        assert route.allocation == "information"
        assert len(route.edge_weights) == len(route.token_ids)
        assert sum(route.edge_weights) == pytest.approx(1.0)
        assert len(route.allocation_diagnostics) == len(route.token_ids)
        assert route.allocation_diagnostics[-1][2] == pytest.approx(1.0)
        assert min(route.edge_weights) >= 0.05

    restored = BiasCatalog.from_json(
        compile_catalog(
            {
                "defaults": {
                    "cases": ["original"],
                    "leading_space": False,
                    "plural": False,
                },
                "terms": [{"shadowing": {"allocation": "information"}}],
            },
            backend,
        ).to_json()
    )
    restored_route = restored.require("shadowing").routes[0]
    assert restored_route.allocation == "information"
    assert restored_route.edge_weights


def test_information_allocation_floor_prevents_zero_head_weight(backend):
    stats = build_prefix_reference_stats({"shadow": 100.0, "shadowing": 1.0})
    weights, diagnostics = allocate_route(
        (1, 8),
        backend,
        strategy="information",
        reference_stats=stats,
        allocation_floor=0.05,
    )

    assert min(weights) >= 0.05
    assert sum(weights) == pytest.approx(1.0)
    assert diagnostics[0][3] == pytest.approx(weights[0])


def test_naive_chaining_allocates_deterministic_route_steps(backend):
    entry = compile_term(
        "port of call",
        "port of call",
        backend,
        options=CompileOptions(
            cases=("original",),
            leading_space=False,
            plural=False,
            allocation="naive_chaining",
            max_route_tokens=3,
        ),
    )

    route = next(route for route in entry.routes if route.token_ids == (17, 18, 19))
    assert "naive_chaining" in ALLOCATIONS
    assert route.allocation == "naive_chaining"
    assert route.edge_weights == pytest.approx((0.0, 0.5, 1.0))
    assert sum(route.edge_weights) == pytest.approx(1.5)
    assert tuple(row[3] for row in route.allocation_diagnostics) == pytest.approx(
        route.edge_weights
    )


def test_external_weighted_reference_replaces_tokenizer_reference(backend):
    catalog = compile_catalog(
        {
            "defaults": {
                "cases": ["original"],
                "leading_space": False,
                "plural": False,
                "allocation": "information",
                "allocation_floor": 0,
            },
            "terms": ["shadowing"],
        },
        backend,
        reference={"shadowing": 1, "unrelated": 100},
    )

    route = next(
        route for route in catalog.require("shadowing").routes
        if route.token_ids == (1, 8)
    )
    # The external lexicon contains only one surface beginning with shadow;
    # tokenizer entries such as shadow and shadowing's vocabulary token are not
    # silently added to the weighted universe.
    assert route.allocation_diagnostics[0][0] == pytest.approx(1.0)
    assert route.allocation_diagnostics[-1][0] == pytest.approx(1.0)


def test_explicit_reference_routes_are_embedded_for_online_prior(backend):
    catalog = compile_catalog(
        {
            "defaults": {
                "cases": ["original"],
                "leading_space": False,
                "plural": False,
            },
            "terms": ["shadowing"],
        },
        backend,
        reference={"shadowing": 10},
    )

    routes = {
        (route.text, route.token_ids): route.weight
        for route in catalog.reference_prior_routes
    }
    assert routes["shadowing", (6,)] == pytest.approx(5.0)
    assert routes[" shadowing", (7,)] == pytest.approx(5.0)
    restored = BiasCatalog.from_json(catalog.to_json())
    assert restored.reference_prior_routes == catalog.reference_prior_routes


def test_accumulated_information_amplifies_only_long_routes(backend):
    stats = build_prefix_reference_stats({
        "shadow": 100,
        "shadowing": 1,
        "shark": 25,
    })
    base_two, _ = allocate_route(
        (1, 8), backend, strategy="information", reference_stats=stats,
        allocation_floor=0,
    )
    amplified_two, _ = allocate_route(
        (1, 8), backend, strategy="information_amplified", reference_stats=stats,
        allocation_floor=0,
    )
    base_three, base_diagnostics = allocate_route(
        (9, 10, 8), backend, strategy="information", reference_stats=stats,
        allocation_floor=0,
    )
    amplified_three, amplified_diagnostics = allocate_route(
        (9, 10, 8), backend, strategy="information_amplified",
        reference_stats=stats, allocation_floor=0,
    )

    assert amplified_two == pytest.approx(base_two)
    assert amplified_three == pytest.approx(tuple(
        weight * (1.0 + diagnostic[2])
        for weight, diagnostic in zip(base_three, base_diagnostics)
    ))
    assert tuple(row[2] for row in amplified_diagnostics) == pytest.approx(
        tuple(row[2] for row in base_diagnostics)
    )
    assert sum(base_three) == pytest.approx(1.0)
    assert sum(amplified_three) > 1.0


def test_group_cycles_and_reserved_global_are_rejected(backend):
    with pytest.raises(EditorError, match="cycle"):
        compile_catalog({"groups": {"a": ["b"], "b": ["a"]}}, backend)
    with pytest.raises(EditorError, match="reserved"):
        compile_catalog({"terms": ["global"]}, backend)


def test_catalog_round_trip_and_merge(backend):
    source = {"defaults": {"cases": ["original"], "plural": False}, "terms": ["shadow"]}
    catalog = compile_catalog(source, backend)
    loaded = BiasCatalog.from_json(catalog.to_json())
    assert loaded.to_dict() == catalog.to_dict()
    merged = BiasCatalog.merge([catalog, loaded])
    assert len(merged.require("shadow").routes) == len(catalog.require("shadow").routes)

    other = compile_catalog({"defaults": {"cases": ["original"], "plural": False},
                             "terms": ["hearth"]}, backend,
                            provenance={"backend": "other", "vocabulary_size": backend.vocabulary_size()})
    with pytest.raises(EditorError, match="different tokenizers"):
        BiasCatalog.merge([catalog, other])


def test_catalog_cli_can_compile_an_inline_term(tmp_path, backend):
    output = tmp_path / "catalog.json"
    with patch("trajectory_editor.bias_cli._load_backend", return_value=backend):
        assert bias_main([
            "--term", "shadow",
            "--level", "minimal",
            "--output", str(output),
        ]) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["format"] == "spe-bias-catalog-v1"
    assert "shadow" in value["entries"]


def test_catalog_cli_exposes_route_policy_override(tmp_path, backend):
    output = tmp_path / "catalog.json"
    with patch("trajectory_editor.bias_cli._load_backend", return_value=backend):
        assert bias_main([
            "--term", "shadow",
            "--level", "exhaustive",
            "--route-policy", "cohesive",
            "--min-route-piece-chars", "4",
            "--output", str(output),
        ]) == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["compiler"]["default_route_policy"] == "cohesive"
    assert value["compiler"]["default_min_route_piece_chars"] == 4


def test_catalog_cli_exposes_information_allocation_diagnostics(tmp_path, backend, capsys):
    output = tmp_path / "catalog.json"
    source = tmp_path / "terms.yaml"
    source.write_text(
        "defaults:\n"
        "  cases: [original]\n"
        "  leading_space: false\n"
        "  plural: false\n"
        "terms: [shadowing]\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference.yaml"
    reference.write_text("shadow: 100\nshadowing: 2\n", encoding="utf-8")
    with patch("trajectory_editor.bias_cli._load_backend", return_value=backend):
        assert bias_main([
            "--input", str(source),
            "--level", "exhaustive",
            "--allocation", "information",
            "--reference", str(reference),
            "--diagnostics",
            "--output", str(output),
        ]) == 0

    value = json.loads(output.read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    assert value["compiler"]["default_allocation"] == "information"
    assert "remaining_mass" in captured.err
    assert "edge_weight" in captured.err


def test_catalog_cli_reports_cohesive_tail_fallback(tmp_path, capsys):
    output = tmp_path / "catalog.json"
    source = tmp_path / "terms.yaml"
    source.write_text(
        "defaults:\n"
        "  cases: [original]\n"
        "  leading_space: false\n"
        "  plural: false\n"
        "terms: [ajar]\n",
        encoding="utf-8",
    )
    with patch("trajectory_editor.bias_cli._load_backend", return_value=RouteRankingBackend()):
        assert bias_main([
            "--input", str(source),
            "--level", "exhaustive",
            "--route-policy", "cohesive",
            "--output", str(output),
        ]) == 0
    captured = capsys.readouterr()
    assert "no cohesive routes" in captured.err
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["compiler"]["warnings"] == [
        "term 'ajar' has no cohesive routes; using a tail-only fallback",
    ]
