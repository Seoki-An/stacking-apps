# Saved-scene rocking comparison: 4D and 6D contacts

> Historical benchmark: these results used the former pose-noise/clearance policy.
> Current robustness uses velocity disturbances; see
> [the current policy](excavator-simulation.md#placement-robustness-reward).

## Active configuration

Excavator `solvers/simulation.yml` now selects `limit_surface_6d` under
`physics.structural.contact.model`. This covers CPU/CUDA short structural
checks, long validation, and noisy release trials. Settling still uses
`single_point`; physical damping is still zero. Motion, energy, clearance,
and robustness thresholds were not changed for this comparison.
The current excavator profile sets `patch_eps: 0.00003`. The original 4D/6D
comparison below used the native default `0.0001`; its results do not validate
the current intermediate setting.
The smaller `0.00001` value is supported, but its late-scene replay failures
prevented adopting it as the active setting (see the epsilon comparison below).

The common profile loader accepts `single_point`, `limit_surface_4d`, and
`limit_surface_6d`. Tabletop now also selects 6D with `patch_eps: 0.00003`.
These saved scenes only validate excavator behavior; tabletop physical behavior
has not been benchmarked here.

Core's 6D model adds two tipping moments whose limits follow the estimated
elliptical contact patch and normal impulse. These represent center-of-pressure
movement within the patch. Sliding and torsion retain their existing friction
budget. This is not an arbitrary angular damping coefficient or additional
sampled contact points. The original comparison used the native patch defaults
(`patch_eps = 1e-4 m`, `patch_damping = 1e-10`). See
[the core derivation](../../stacking-core/docs/limit-surface-6d.md).

## Comparison on 2026-10-13

Inputs were `excavator_plan1.stack`, steps 1–4, and `excavator_plan2.stack`,
steps 1–6: **655 saved root candidates** with their original parent scenes.
Each candidate was replayed with both short and long profiles and both contact
models. The same updated native binary, asset models, single-point settling,
noise seeds, and current app policies were used in both arms. This isolates the
contact-model change; it is not a comparison against every setting or binary
used when the files were originally saved.

| Result across 655 candidates | 4D short | 6D short | 4D long | 6D long |
|---|---:|---:|---:|---:|
| Nominal simulation settled, before other rejection checks | 294 | 655 | 272 | 655 |
| Accepted after simulation and robustness checks | 185 | 421 | 168 | 421 |
| Primary rejection: `simulation_unsettled` | 177 | 0 | 106 | 0 |
| Primary rejection: `target_motion_limit` | 70 | 0 | 223 | 0 |
| Primary rejection: initial clearance | 160 | 173 | 110 | 173 |
| Primary rejection: robustness excursion | 63 | 61 | 48 | 61 |
| Sum of per-step median evaluation time | 8.210 s | 7.682 s | 28.382 s | 15.722 s |

Every candidate accepted by the 4D replay remained accepted by 6D. Outcomes
were identical across three repetitions. Neither model recorded unconverged
simulation steps. Timing excludes warmup, alternates model order, and uses
placement batches of up to 32; it includes settling, structural validation,
four noisy trials, result conversion, and benchmark motion measurement.
The observed reductions are **6.4% short / 44.6% long**. They do not measure
whole MCTS runtime, whose explored tree changes when more candidates survive.

The increased number of *primary* clearance rejections is failure precedence:
previous nominal motion failures had hidden some clearance failures. In both
arms, exactly **294 of 2,620 noisy trials** failed initial clearance. Under 6D,
2,230 trials settled and 96 exceeded the excursion limit. Under 4D long,
824 settled, 1,347 remained unsettled, 22 had bounded rocking, and 133 exceeded
the excursion limit. No clearance or excursion guard was relaxed.

## Candidate 25: plan1, step 1

| Production replay metric | 4D | 6D |
|---|---:|---:|
| Short-stage result | Motion limit at 1.50 s | Quiet at 0.83 s |
| Long validation result | Short gate rejects | Quiet; total 4.30 s |
| Target accumulated motion | 0.201232 | 0.050417 |
| Final normalized energy | 0.028395 | 2.05e-12 |
| Four noisy trials | All unsettled | All settled |

A CPU replay independently reproduced the same accept/reject change and stage
durations; 6D accumulated motion was 0.050414. These CPU timings were single
calls, not a CPU performance benchmark.

An additional diagnostic started both models from the same post-settling scene
and disabled only the cumulative-motion cutoff to observe the longer motion.
4D still rocked at 15.01 s, with a final one-second position extent of
28.43 mm and angular extent of 6.47 degrees. 6D reached the quiet condition
at 4.50 s with zero measurable pose extent in the final second. This diagnostic
uses one long phase, whereas production carries the short phase's quiet count
into the long phase; that explains 4.50 versus 4.30 seconds.

## Patch epsilon comparison on 2026-10-14

`physics.structural.contact.patch_eps` is now an optional platform setting,
passed to both CPU and CUDA. Omission preserves the core default. The excavator
and tabletop profiles currently use `0.00003`; settling retains its default.
Values must be finite and non-negative.

For a fixed local curvature, the patch axes scale with
`sqrt(max(patch_eps - gap, 0))`. At zero gap, changing `1e-4` to `1e-5` reduces
both axes to about 31.6% and the patch area to 10%. Tipping and torsional moment
capacities decrease with those axes. This changes the assumed contact support;
it does not tighten the solver tolerance or collision detection margin. The
ratio differs when the contact has nonzero gap or penetration.

The original 655 candidates were replayed with both epsilons, three repetitions,
and both fidelities. Outcomes were deterministic across repetitions.

| Result | `1e-4` | `1e-5` |
|---|---:|---:|
| Accepted, short and long | 421 / 655 | 411 / 655 |
| Nominal simulations settled | 655 | 640 |
| Primary target-motion rejections | 0 | 15 |
| Sum of per-step median short evaluation time | 7.807 s | 9.505 s |
| Sum of per-step median long evaluation time | 15.874 s | 18.055 s |

Eleven accepted candidates became rejected and one previously rejected
candidate became accepted. No originally selected candidate in these ten steps
was among those eleven. Smaller epsilon increased evaluation time by 21.7%
for short and 13.7% for long checks. No unconverged steps occurred in this set.
Candidate 25 still passed on CPU and CUDA, with four settled noisy trials:
its short check increased from 0.83 to 1.45 simulated seconds, while full
validation remained 4.30 seconds. Its motion integral increased from 0.0504
to 0.0863, below the existing 0.2 threshold.

The crowded final scenes of `excavator_plan3.stack` gave a different result:

| Long replay | `1e-4` accepted | `1e-5` accepted |
|---|---:|---:|
| Step 32, 32 candidates | 15 | 0 |
| Step 33, 29 candidates | 7 | 0 |

At `1e-5`, all 61 candidates hit scene-motion (33) or target-motion (28)
limits. Total unconverged simulation steps also increased from 5 to 120,
so numerical convergence needs checking alongside the weaker support moments.
This late-scene check used one measured repetition per epsilon; its timings
are not a repeat-based performance benchmark.

Consequently, **the active epsilon remains `1e-4`**. The profile can request
`1e-5`, but the saved late stack was constructed under `1e-4`; changing its
support model during replay is not equivalent to building a fresh stack with
`1e-5` throughout. A fresh run and physical support calibration are needed
before recommending the smaller value. More accepted candidates alone do not
establish which contact model is physically accurate.

The [epsilon comparison report](benchmarks/excavator-contact-patch-eps-20261014.json)
records counts, changed candidate IDs, timings, profiles, CPU candidate 25
diagnostics, and hashes. Detailed local reports are
`.benchmarks/contact-patch-eps-20261014.json` and
`.benchmarks/contact-patch-eps-late-20261014.json`. Input stack hashes were
verified unchanged. Reproduce with:

```bash
python tests/benchmark_contact_models.py \
  --stacks .stacks/excavator_plan1.stack .stacks/excavator_plan2.stack \
  --patch-eps 1e-4 1e-5 --output /tmp/contact-patch-eps.json
python tests/benchmark_contact_models.py \
  --stacks .stacks/excavator_plan3.stack --steps 32 33 \
  --fidelities long --repeats 1 --patch-eps 1e-4 1e-5 \
  --output /tmp/contact-patch-eps-late.json
```

## Reproduction and artifacts

The tested CUDA build exposes `ContactModel.LIMIT_SURFACE_6D`. The installed
extension had been stale; the tested `build-cuda` extension was activated by
atomic replacement, preserving the old binary at
`/tmp/stacking-core-native-before-6d-c2b41d436e41.so`. New processes use 6D;
restart already-open planning/operations processes. No core source was changed.

For a clean installation, rebuild using the core's supported installation:

```bash
STACKING_CORE_ENABLE_CUDA=ON python -m pip install -e ../stacking-core/python
python tests/benchmark_contact_models.py \
  --stacks .stacks/excavator_plan1.stack .stacks/excavator_plan2.stack \
  --output /tmp/contact-models.json
```

[The compact recorded report](benchmarks/excavator-contact-models-20261013.json)
contains input/native/script hashes, profile values, all per-step timing and
outcome counts, transition counts, and candidate 25 diagnostics. The full
per-candidate replay is retained locally at
`.benchmarks/contact-models-20261013.json`; the benchmark command regenerates
that detailed format. Input `.stack` hashes were verified unchanged.

These results establish a substantial improvement for the saved simulation
cases. All nominal candidates settling does not independently establish real
stone stability: the estimated contact-patch moment capacity still needs
comparison with physical stable and tipping examples. A full new stacking run
and tabletop saved-scene comparison were not part of this replay. Existing
`.stack` records keep their original outcomes; new planning uses the new model.
