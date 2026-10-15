"""Conservative architecture discovery and parameter-identity training selection."""
from dataclasses import dataclass
import re
from .core.errors import EditorError

BLOCKS = ('lm_head', 'embeddings', 'attention', 'attention_qk', 'attention_vo',
          'attention_other', 'mlp', 'norm', 'other')
ALIASES = {'im_head': 'lm_head', 'q_proj': 'attention_qk', 'k_proj': 'attention_qk',
           'v_proj': 'attention_vo', 'o_proj': 'attention_vo'}


def block_for(name):
    parts = name.lower().split('.')
    if any(p in ('lm_head', 'embed_out', 'output', 'output_projection') for p in parts):
        return 'lm_head'
    if any(p in ('embed_tokens', 'embed_positions', 'wte', 'wpe', 'word_embeddings',
                 'position_embeddings', 'tok_embeddings') for p in parts):
        return 'embeddings'
    if any('norm' in p or p in ('ln_1', 'ln_2', 'ln_f') for p in parts):
        return 'norm'
    if any(p in ('mlp', 'feed_forward', 'ffn') for p in parts):
        return 'mlp'
    if any(p in ('attn', 'self_attn', 'attention', 'self_attention') for p in parts):
        if any(p in ('q_proj', 'k_proj', 'query', 'key', 'wq', 'wk') for p in parts):
            return 'attention_qk'
        if any(p in ('v_proj', 'o_proj', 'out_proj', 'c_proj', 'value', 'wv', 'wo') for p in parts):
            return 'attention_vo'
        return 'attention_other'  # Includes fused QKV: cannot split a parameter.
    return 'other'


def layer_for(name):
    match = re.search(r'(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)(?:\.|$)', name)
    return int(match[1]) if match else None


@dataclass
class Target:
    parameter: object
    names: tuple
    blocks: frozenset
    layers: frozenset


class LearningTargets:
    def __init__(self, model):
        grouped = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            entry = grouped.setdefault(id(parameter), [parameter, []])
            entry[1].append(name)
        embedding_ids = {}
        for getter, block in (("get_input_embeddings", "embeddings"), ("get_output_embeddings", "lm_head")):
            method = getattr(model, getter, None)
            module = method() if callable(method) else None
            if module is not None:
                for parameter in module.parameters():
                    embedding_ids.setdefault(id(parameter), set()).add(block)
        self.targets = [Target(p, tuple(names),
                               frozenset(embedding_ids.get(id(p), {block_for(n) for n in names})),
                               frozenset(layer_for(n) for n in names))
                        for p, names in grouped.values()]
        self.layers = sorted({n for t in self.targets for n in t.layers if n is not None})

    def inventory(self):
        return dict(layers=self.layers, supported_blocks=list(BLOCKS), aliases=ALIASES,
                    parameters=[dict(names=list(t.names), blocks=sorted(t.blocks),
                                     layers=sorted(n for n in t.layers if n is not None),
                                     global_parameter=None in t.layers,
                                     elements=t.parameter.numel()) for t in self.targets],
                    note='Tied aliases are one parameter. Fused QKV is attention_other; use attention to include it.')

    def select(self, config):
        names = set(config.freeze or config.thaw)
        names = {ALIASES.get(n, n) for n in names}
        unknown = names - set(BLOCKS)
        if unknown:
            raise EditorError('unknown learning blocks: ' + ', '.join(sorted(unknown)))
        available = {b for t in self.targets for b in t.blocks}
        if any(b != 'attention' and b not in available for b in names) or ('attention' in names and not any(b.startswith('attention_') for b in available)):
            raise EditorError('learning block absent from model; inspect --target-list')
        if 'attention' in names:
            names.remove('attention')
            names.update(('attention_qk', 'attention_vo', 'attention_other'))
        layers = None
        if config.layers:
            layers = set()
            for term in config.layers:
                if re.fullmatch(r'\d+', term):
                    chosen = {int(term)}
                elif re.fullmatch(r'\d+-\d+', term):
                    a, b = map(int, term.split('-'))
                    if a > b:
                        raise EditorError('layer range must be ascending')
                    chosen = set(range(a, b + 1))
                elif re.fullmatch(r'\d+\+', term):
                    chosen = {n for n in self.layers if n >= int(term[:-1])}
                else:
                    raise EditorError('layers use indices, inclusive A-B ranges, or N+')
                if not chosen or chosen - set(self.layers):
                    raise EditorError('layer selection outside discovered model layers')
                layers.update(chosen)
        selected = []
        for t in self.targets:
            active = not (t.blocks & names) if config.freeze else (bool(t.blocks & names) if config.thaw else True)
            if active and config.thaw and not t.blocks <= names:
                raise EditorError('thaw selection crosses tied aliases: ' + ', '.join(t.names))
            if active and layers is not None:
                active = None not in t.layers and t.layers <= layers
            if active:
                selected.append(t)
        if not selected:
            raise EditorError('learning selection contains no parameters')
        return selected

    def apply(self, config):
        selected = self.select(config)
        identities = {id(t.parameter) for t in selected}
        for t in self.targets:
            t.parameter.requires_grad_(id(t.parameter) in identities)
            t.parameter.grad = None
        return dict(trainable_elements=sum(t.parameter.numel() for t in selected),
                    trainable_names=[n for t in selected for n in t.names],
                    preserves_kv=all(t.blocks == {'lm_head'} for t in selected))
