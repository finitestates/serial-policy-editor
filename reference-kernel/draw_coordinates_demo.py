"""Reproducible coordinate counterexample; no model or production dependencies."""
import json
from reference_kernel import Distribution, World, draw, position_uniform

world = World.for_prefix(12345, (1, 2, 3))
distribution = Distribution((0, 1, 2, 3), (.25, .25, .25, .25), (0., 0., 0., 0.))
first = draw(distribution, world, 0, 'categorical')
for boundary in range(1, 100):
    token = draw(distribution, world, boundary, 'categorical')
    if token != first:
        break
else:
    raise RuntimeError('no coordinate counterexample found')
print(json.dumps({'seed': world.seed, 'stream_fingerprint': world.stream_fingerprint,
                  'distribution': list(zip(distribution.ids, distribution.probabilities)),
                  'draws': [{'boundary': step, 'uniform': position_uniform(world, step),
                             'token_id': draw(distribution, world, step, 'categorical')}
                            for step in (0, boundary)],
                  'explanation': 'Same logits, policy, seed and root; changing only the boundary changes the proposal.'}, indent=2))
