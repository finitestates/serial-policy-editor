"""Static hot-path rule checker.

Parses the trajectory_editor package, builds a conservative call/reference
graph from declared interactive roots, and reports every reference to a
banned symbol that is reachable from those roots. The checker is
deliberately over-approximate:

* It counts *references*, not just calls, so ``f = self.backend.reset; f()``
  and ``callback=engine.observe`` are caught.
* Nested functions and lambdas count as part of the enclosing function.
* ``getattr(x, "name")`` with a literal name counts as a reference to ``name``.
* Import aliases are resolved (``from copy import deepcopy as dc``).

False positives are resolved by an allowlist entry with a justification and
a named test, never by weakening the checker.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Ref:
    name: str            # attribute or resolved qualified name ("copy.deepcopy")
    receiver: str        # "backend" | "engine" | "self" | "module" | "other"
    line: int
    in_swallow: bool     # inside try-body whose handler catches Exception/BaseException


@dataclass
class Func:
    key: str             # "module:Class.method" or "module:function"
    module: str
    cls: str | None
    node: ast.AST
    refs: list[Ref] = field(default_factory=list)
    edges: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Violation:
    rule: str
    function: str
    symbol: str
    line: int
    via: tuple[str, ...]  # call chain from a root

    def key(self) -> tuple[str, str, str]:
        return (self.rule, self.function, self.symbol)

    def __str__(self) -> str:
        chain = " -> ".join(self.via)
        return f"[{self.rule}] {self.function}:{self.line} references {self.symbol!r}\n    via {chain}"


_SWALLOWING = {"Exception", "BaseException"}


def _catches_everything(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(t, ast.Name) and t.id in _SWALLOWING for t in types)


def _last_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):  # super().x, factory().x
        return _last_name(node.func)
    return None


class Package:
    def __init__(self, root: Path, rules) -> None:
        self.root = root
        self.rules = rules
        self.funcs: dict[str, Func] = {}
        self.class_bases: dict[str, list[str]] = {}      # "mod:Class" -> ["mod:Base"]
        self.methods_by_class: dict[str, set[str]] = {}
        self.imports: dict[str, dict[str, str]] = {}     # module -> alias -> qualified
        self.module_funcs: dict[str, set[str]] = {}
        for path in sorted(root.rglob("*.py")):
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            self._index(module, ast.parse(path.read_text(), str(path)))
        for func in self.funcs.values():
            self._collect(func)

    # ---- indexing -------------------------------------------------------
    def _index(self, module: str, tree: ast.Module) -> None:
        imports: dict[str, str] = {}
        package_prefix = module.rsplit(".", 1)[0] + "." if "." in module else ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = module.split(".")[: -node.level]
                    base = ".".join([*parts, base] if base else parts)
                    base = "pkg:" + base
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{base}.{alias.name}"
        self.imports[module] = imports
        self.module_funcs[module] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = f"{module}:{node.name}"
                self.funcs[key] = Func(key, module, None, node)
                self.module_funcs[module].add(node.name)
            elif isinstance(node, ast.ClassDef):
                ckey = f"{module}:{node.name}"
                self.class_bases[ckey] = [
                    self._resolve_class(module, _last_name(b) or "") for b in node.bases
                ]
                self.methods_by_class[ckey] = set()
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        key = f"{ckey}.{item.name}"
                        self.funcs[key] = Func(key, module, node.name, item)
                        self.methods_by_class[ckey].add(item.name)

    def _resolve_class(self, module: str, name: str) -> str:
        target = self.imports.get(module, {}).get(name)
        if target and target.startswith("pkg:"):
            mod, _, cls = target[4:].rpartition(".")
            return f"{mod}:{cls}"
        return f"{module}:{name}"

    # ---- receiver classification ---------------------------------------
    def _receiver_kind(self, func: Func, value: ast.AST) -> str:
        name = _last_name(value)
        owner = f"{func.module}:{func.cls}" if func.cls else None
        if isinstance(value, ast.Name) and value.id == "self" and owner:
            for kind, classes in self.rules.CLASS_KINDS.items():
                if owner in classes:
                    return kind
            return "self"
        for kind, names in self.rules.RECEIVER_KINDS.items():
            if name in names:
                return kind
        return "other"

    def _method_targets(self, func: Func, value: ast.AST, attr: str) -> list[str]:
        """Package functions an attribute reference may resolve to."""
        owner = f"{func.module}:{func.cls}" if func.cls else None
        classes: list[str] = []
        if isinstance(value, ast.Name) and value.id in {"self", "cls"} and owner:
            classes = [owner]
        elif isinstance(value, ast.Call) and _last_name(value.func) == "super" and owner:
            classes = list(self.class_bases.get(owner, []))
        else:
            name = _last_name(value)
            classes = list(self.rules.RECEIVER_TYPES.get(name, ()))
            if isinstance(value, ast.Name):
                target = self.imports.get(func.module, {}).get(value.id, "")
                if target.startswith("pkg:"):  # module alias or imported class
                    mod, _, leaf = target[4:].rpartition(".")
                    if f"{mod}:{leaf}" in self.methods_by_class:
                        classes.append(f"{mod}:{leaf}")
                    elif f"{target[4:]}:{attr}" in self.funcs:
                        return [f"{target[4:]}:{attr}"]
        found = []
        seen = set()
        while classes:
            cls = classes.pop()
            if cls in seen:
                continue
            seen.add(cls)
            if attr in self.methods_by_class.get(cls, ()):
                found.append(f"{cls}.{attr}")
            else:
                classes.extend(self.class_bases.get(cls, []))
        return found

    # ---- reference collection ------------------------------------------
    def _collect(self, func: Func) -> None:
        swallow_lines: set[int] = set()
        for node in ast.walk(func.node):
            if isinstance(node, ast.Try) and any(_catches_everything(h) for h in node.handlers):
                for stmt in node.body:
                    for sub in ast.walk(stmt):
                        if hasattr(sub, "lineno"):
                            swallow_lines.add(sub.lineno)
        imports = self.imports[func.module]
        for node in ast.walk(func.node):
            line = getattr(node, "lineno", 0)
            swallowed = line in swallow_lines
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                kind = self._receiver_kind(func, node.value)
                if isinstance(node.value, ast.Name) and node.value.id in imports:
                    qualified = imports[node.value.id]
                    if not qualified.startswith("pkg:"):
                        func.refs.append(Ref(qualified, "module", line, swallowed))
                        func.refs.append(Ref(f"{qualified}.{node.attr}", "module", line, swallowed))
                func.refs.append(Ref(node.attr, kind, line, swallowed))
                func.edges.update(self._method_targets(func, node.value, node.attr))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                target = imports.get(node.id)
                if target and not target.startswith("pkg:"):
                    func.refs.append(Ref(target, "module", line, swallowed))
                elif target and target.startswith("pkg:"):
                    mod, _, leaf = target[4:].rpartition(".")
                    if f"{mod}:{leaf}" in self.funcs:
                        func.edges.add(f"{mod}:{leaf}")
                    if f"{mod}:{leaf}" in self.methods_by_class:
                        func.edges.add(f"{mod}:{leaf}.__init__")
                elif node.id in self.module_funcs[func.module]:
                    func.edges.add(f"{func.module}:{node.id}")
                elif f"{func.module}:{node.id}" in self.methods_by_class:
                    func.edges.add(f"{func.module}:{node.id}.__init__")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                attr = node.args[1].value
                func.refs.append(Ref(attr, self._receiver_kind(func, node.args[0]), line, swallowed))
                func.edges.update(self._method_targets(func, node.args[0], attr))
        func.edges &= set(self.funcs)

    # ---- analysis -------------------------------------------------------
    def closure(self, roots, stop: set[str] = frozenset()) -> dict[str, tuple[str, ...]]:
        """Reachable function -> shortest chain from a root.

        Edges into functions whose short name is in ``stop`` are not followed;
        the reference to them is reported at the caller instead.
        """
        missing = [r for r in roots if r not in self.funcs]
        if missing:
            raise AssertionError(f"declared roots do not exist: {missing}")
        chains = {root: (root,) for root in roots}
        frontier = list(roots)
        while frontier:
            nxt = []
            for key in frontier:
                for edge in sorted(self.funcs[key].edges):
                    short = edge.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
                    if short in stop:
                        continue
                    if edge not in chains:
                        chains[edge] = (*chains[key], edge)
                        nxt.append(edge)
            frontier = nxt
        return chains

    def check(self) -> list[Violation]:
        rules = self.rules
        out: list[Violation] = []

        def matches(ref: Ref, symbols) -> str | None:
            for kind, name in symbols:
                if ref.name == name and (kind == "any" or kind == ref.receiver):
                    return name
            return None

        interactive = self.closure(rules.INTERACTIVE_ROOTS)
        for key, chain in interactive.items():
            for ref in self.funcs[key].refs:
                symbol = matches(ref, rules.NEVER_ON_INTERACTIVE_PATH)
                if symbol:
                    out.append(Violation("never-on-interactive-path", key, symbol, ref.line, chain))
                symbol = matches(ref, rules.FULL_PREFILL)
                if symbol:
                    out.append(Violation("no-full-prefill", key, symbol, ref.line, chain))
                if ref.in_swallow and (
                    ref.receiver == "backend"
                    or matches(ref, rules.NEVER_ON_INTERACTIVE_PATH)
                ):
                    out.append(Violation("no-swallowed-backend-errors", key, ref.name, ref.line, chain))

        for rule, roots, symbols in (
            ("invalidate-dont-refresh", rules.NO_EAGER_READS, rules.EAGER_READS),
            ("pure-bookkeeping", rules.PURE_BOOKKEEPING, rules.ANY_BACKEND_WORK),
        ):
            stop = {name for _, name in symbols}
            for key, chain in self.closure(roots, stop).items():
                for ref in self.funcs[key].refs:
                    symbol = matches(ref, symbols)
                    if symbol:
                        out.append(Violation(rule, key, symbol, ref.line, chain))

        # One violation per (rule, function, symbol); keep the first line.
        unique: dict[tuple, Violation] = {}
        for v in sorted(out, key=lambda v: (v.key(), v.line)):
            unique.setdefault(v.key(), v)
        return list(unique.values())
