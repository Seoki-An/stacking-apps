# Tabletop height-map benchmark

This benchmark compares the dependency-free NumPy rasterizer used by
`TabletopHeightMapRenderer` with the legacy Open3D downward ray-cast. Both use
the same commissioned convex-component meshes, identity object poses, a
128-by-128 output, and 20 timed repetitions after one warm-up call.

```shell
python apps/tabletop/scripts/benchmark_heightmap.py \
  --repository . \
  --model-directory /path/to/commissioned/objects \
  --object-ids 1 2 3 4 5 6 7 8 9 10 \
  --repeats 20
```

| Objects | NumPy median | Open3D median | NumPy / Open3D | Maximum error | Occupancy mismatch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.294 ms | 0.292 ms | 7.867 | 0.269 um | 0 / 16,384 |
| 10 | 20.558 ms | 0.606 ms | 33.941 | 0.216 um | 0 / 16,384 |

The numerical outputs have no practically significant discrepancy in these
cases, but the NumPy implementation is not a performance replacement for
Open3D. These measurements are a development snapshot, not a performance
contract. Repeat them on the deployment machine before choosing the production
renderer.
