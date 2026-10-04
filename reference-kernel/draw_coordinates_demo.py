"""Reproducible coordinate counterexample; no model or production dependencies."""
import json
from reference_kernel import Distribution, World, draw, position_uniform_token

world = World.for_prefix(12345, (1, 2, 3))
distribution = Distribution((0, 1, 2, 3), (0.0, 0.0, 0.0, 0.0))
first = draw(distribution, world, 0, 'gumbel-max')
for boundary in range(1, 100):
    token = draw(distribution, world, boundary, 'gumbel-max')
    if token != first:
        break
else:
    raise RuntimeError('no coordinate counterexample found')
print(json.dumps({'seed': world.seed, 'stream_fingerprint': world.stream_fingerprint,
                  'eligible_scores': list(zip(distribution.ids, distribution.scores)),
                  'draws': [{'boundary': step, 'token_uniforms': [position_uniform_token(world, step, token_id) for token_id in distribution.ids],
                             'token_id': draw(distribution, world, step, 'gumbel-max')}
                            for step in (0, boundary)],
                  'explanation': 'Same logits, policy, seed and root; changing only the boundary changes the proposal.'}, indent=2))
