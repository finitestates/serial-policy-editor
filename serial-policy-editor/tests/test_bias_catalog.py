import json
from pathlib import Path
from unittest.mock import patch

import pytest

from trajectory_editor.bias_catalog import (
    BiasCatalog,
    CompileOptions,
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
    with pytest.raises(EditorError, match="term 'Shadow'.*max_routes=1"):
        compile_term(
            "Shadow",
            "Shadow",
            backend,
            options=CompileOptions(
                level="minimal", cases=("original", "lower"),
                leading_space=False, plural=False, max_routes=1,
            ),
        )


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
