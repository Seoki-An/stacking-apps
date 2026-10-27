# MCTS reward and initial pose sampling

Paper reference for RSS 2027. Reviewed on 2026-10-26 against the working tree.

The method combines geometry-guided placement proposals, continuous pose
optimization, and simulation-based evaluation within MCTS. Proposal scores,
optimization-derived action priors, and transition rewards serve different roles.

## 1. Reward design

Let action $a$ place stone $i$, yielding nominal simulated successor $s'$.
For the initial inventory of $M$ stones, define fixed normalization scales

$$
\bar V=\frac{1}{M}\sum_{i=1}^{M}V_i,
\qquad L=\bar V^{1/3}.
$$

The current excavator reward is

$$
r(s,a,s')=
\phi_{\mathrm{volume}}+2\phi_{\mathrm{contain}}
-2\psi_{\mathrm{equilibrium}}-2\psi_{\mathrm{flatness}}
-2\psi_{\mathrm{disturbance}}.
$$

Tabletop additionally enables an inward-orientation bonus with weight $2$.
The large-object-at-low-height reward has weight zero on both platforms.
The total reward is not clipped. Rejection checks are separate from this sum.

### Target volume and containment

For placed geometry $B_i(s')$ and target $T$, voxel intersection estimates give

$$
C_i=\operatorname{clip}\left(
\frac{\widehat{\operatorname{Vol}}(B_i(s')\cap T)}{V_i},0,1\right),
\qquad
\phi_{\mathrm{contain}}=\max(C_i-0.5,0).
$$

For the standard single target,

$$
\phi_{\mathrm{volume}}=\frac{C_iV_i}{\bar V}.
$$

A fully contained mean-sized stone contributes one unit of volume reward.
The diagnostic names containing `IoU` are historical: these quantities are
containment and coverage, rather than intersection-over-union. Previously placed
stones are evaluated at their successor poses, so this term does not subtract
coverage lost through their motion from the parent state.

The current working-tree composite evaluator instead counts newly covered union
voxels. With $U_-$ the union of earlier stones at their successor poses,

$$
\phi_{\mathrm{volume}}=
\frac{\widehat{\operatorname{Vol}}\!\left(
[B_i(s')\cap T]\setminus U_-\right)}{\bar V}.
$$

This avoids counting already occupied target voxels twice.

### Equilibrium penalty

Using residual forces and torques returned by pose optimization,

$$
E=\max_b\left[
\frac{\|\mathbf f_b\|_2}{m_bg}
+\frac{\|\boldsymbol\tau_b\|_2}{m_bgV_b^{1/3}}
\right],
\qquad
\psi_{\mathrm{equilibrium}}=\operatorname{clip}(E,0,1).
$$

The maximum is over participating bodies reported by the native solver.
This diagnostic comes from pose optimization; it is not recomputed by an
equilibrium solve at the simulated successor. Feasibility filtering retains a
separate legacy equilibrium metric, so its threshold should not be identified
with this dimensionless residual.

### Surface relief penalty

Let $h_{s'}(x,y)$ be the upper surface above the physical base over footprint
$\Omega$, with zero height at empty locations. The continuous interpretation is

$$
\psi_{\mathrm{flatness}}=
\frac{\displaystyle\max_{(x,y)\in\Omega}h_{s'}(x,y)
-\displaystyle\frac{1}{|\Omega|}\int_\Omega h_{s'}(x,y)\,dx\,dy}{L}.
$$

The standard target uses a $32\times32$ grid over its rectangular base.
The working-tree composite implementation uses a separate masked grid per
connected region and cell-area-weighted averaging over the union footprint.
Empty target cells contribute zero; gaps outside the composite footprint are
excluded. This measures maximum-to-mean relief, including incomplete coverage,
rather than height variance or local surface-normal flatness.

### Disturbance robustness

Trials start from the nominal settled scene. Only the newly placed stone receives
a sampled velocity disturbance; other bodies retain their mobility and begin
with zero residual velocity. For trial $k$, dynamic body $b$, and time $t$,

$$
d_{k,b}(t)=\|\mathbf p_{k,b}(t)-\mathbf p_b^0\|_2
+\lambda_\theta d_{\mathrm{SO}(3)}(R_{k,b}(t),R_b^0),
$$

$$
D=\max_k\max_b\max_t d_{k,b}(t),
\qquad
\psi_{\mathrm{disturbance}}=\frac{\widetilde D}{D_{\mathrm{ref}}}.
$$

Here $\widetilde D$ applies the configured nonfinite replacement and cap.
This measures worst transient excursion across trials and dynamic bodies,
including disturbances propagated to earlier stones. Failure to recover or
excessive excursion can also reject a placement independently of the reward.

| Setting | Excavator | Tabletop |
|---|---:|---:|
| Trials | 4 | 4 |
| Gaussian linear velocity standard deviation, each world axis | 0.05 m/s | 0.02 m/s |
| Gaussian angular velocity standard deviation, each body axis | 5 degrees/s | 5 degrees/s |
| Rotation weight $\lambda_\theta$ | 0.1 m/rad | 0.01 m/rad |
| Reference displacement and excursion rejection limit | 0.15 m | 0.015 m |

The profiles mark disturbance magnitudes as provisional. Report the actual
experimental settings rather than implying calibrated release uncertainty.

## 2. Initial pose proposals

### Stone and orientation hypotheses

Fresh proposal budgets are volume-biased, $p(i)\propto V_i^\alpha$, with
$\alpha=1$ and a per-stone attempt cap.

- Excavator aligns principal axes with the longest axis along $X$ and shortest
  along $Z$, then combines two rolls (0 and 180 degrees) with four cardinal yaws.
- Tabletop preserves measured tilt, yaw-aligns the projected longest axis, and
  considers two headings separated by 180 degrees.

These discrete orientations initialize continuous pose optimization.

### Geometric score

The standard sampler evaluates a $32\times32$ planar grid using

$$
S(x,R)=-3H(x,R)+2Q(x,R)+2F(x,R)+1.5B(x,R).
$$

| Term | Meaning |
|---|---|
| $H$ | Normalized mean height of predicted contact locations; favors low support. |
| $Q$ | Fraction of projected bottom samples close to contact; favors broad support. |
| $F$ | Normalized unfilled active-layer area beneath the candidate footprint. |
| $B$ | Proximity to the feasible anchor-region boundary; favors perimeter construction. |

For projected footprint $A_R$, rotated bottom height $b_R(u)$, and scene surface
$h(x+u)$, the support estimate has the continuous interpretation

$$
z^*(x,R)=\max_{u\in A_R}[h(x+u)-b_R(u)],
$$

$$
\mathcal C(x,R)=\{u\in A_R:
z^*(x,R)+b_R(u)-h(x+u)\le\epsilon_c\},
$$

$$
Q(x,R)=\frac{1}{|A_R|}\int_{A_R}\mathbf1_{\mathcal C(x,R)}(u)\,du,
\qquad
\bar h_c(x,R)=\frac{\int_{\mathcal C(x,R)}h(x+u)\,du}{|\mathcal C(x,R)|},
$$

$$
H=\frac{\bar h_c-h_{\min}}{\max(h_{\max}-h_{\min},10^{-6})}.
$$

The implementation uses discrete samples of the overall stone convex hull's
bottom surface. Height extrema are taken over score-map hypotheses. The fill
term uses a scaled, orientation-dependent rectangular footprint approximation.
For the standard target, boundary proximity is normalized Chebyshev distance
from the feasible anchor-box center, with bounds adjusted for target taper.

### Layer policy and draws

1. Seed an empty scene near feasible target corners, using full-footprint bounds.
2. Prioritize the lowest incomplete layer. A $16\times16$ projected coverage
   estimate advances the layer at 75% coverage for excavator and 65% for tabletop.
3. Apply connectivity and unfilled-area rules to encourage connected growth.
4. Within ordinary candidate pools, select approximately 25% elites, then draw
   without replacement using $p_j\propto\exp(S_j/0.3)$.

The optimized action is also checked against the active-layer policy.

### Height initialization and refinement

After choosing $x,y,R$, initialize above overlapping geometry:

$$
z_0=h_{\mathrm{overlapping\ geometry}}+\delta
-\min_{v\in B_i}(Rv)_z.
$$

Clearance $\delta$ is 0.15 m for excavator and 0.015 m for tabletop. Excavator
uses reconstructed PLY vertices when available; tabletop uses inventory mesh
geometry. This lifted seed differs from the contact height used for scoring.
Candidates undergo staged pose optimization and feasibility filtering; CUDA
batches independent optimization jobs.

## 3. Root refinement and tree priors

The pre-simulation prior logit uses the optimized pose:

$$
\ell(a)=-2\operatorname{clip}(E(a),0,1)+2C_i^{\mathrm{pre}}(a),
\qquad
P(a\mid s)=\frac{e^{\ell(a)}}{\sum_b e^{\ell(b)}}.
$$

Softmax is taken over the relevant selectable children. At the root, a CEM-style
procedure ranks proposals with

$$
J_{\mathrm{CEM}}(a)=\ell(a)+\widehat S(a),\qquad\widehat S\in[0,1].
$$

It uses a population of 96, 32 elites, three rounds, and up to 32 retained
proposals. Later rounds mix 50% fresh proposals with Gaussian XY/yaw mutations,
followed by renewed initialization and pose optimization. Mutation scales are
fixed: XY standard deviation 0.2 m for excavator and 0.01 m for tabletop, with
5-degree yaw standard deviation on both. Retention also limits proposals per
stone to two.

The proposal score $S$, prior logit $\ell$, and simulated reward $r$ should be
defined separately in the paper.

## 4. Composite targets and current scope

The composite sampler groups touching ground sections into connected regions,
uses a local grid and active layer per region, and allocates samples round-robin
across regions. Component containment masks exclude gaps and concave notches.
Opening seeds use feasible directional extrema.

Its score differs from the standard sampler: height is normalized by region
target height, and boundary preference uses distance to the feasible-mask
boundary. Under the current 25% elite setting, its one-at-a-time draws always
choose an elite, making selection greedy within each eligible regional pool.

Native pose optimization uses the conservative component containment criterion

$$
g=\max_i\min_{\|u\|=1}
\left[h_{T_i}(u)-\max_j h_{C_j}(u)\right].
$$

It seeks to fit the whole stone hull inside one target component. It does not
establish arbitrary containment spanning several target components.

**Working-tree update since the preceding chat summary:** composite rewards and
observations now have implementations, including union-voxel coverage and masked
surface relief. The full search entry points remain guarded pending viewer and
saved-run support. These changes are uncommitted at this review; this document
records code inspection, not an end-to-end experimental validation.

## 5. Suggested main-paper paragraph

We generate placement hypotheses using geometry-guided sampling over stone
identities, discrete orientations, and planar locations. A layer-filling policy
prioritizes connected growth on the lowest incomplete layer, while geometric
scores favor low support height, broad predicted contact, uncovered target area,
and boundary proximity. Hypotheses are refined by continuous pose optimization,
with additional population-based refinement at the search root. MCTS evaluates
placements using normalized target-contained volume and containment bonuses,
together with penalties for equilibrium residual, surface relief, and worst-case
transient displacement under sampled velocity disturbances. Optimization-derived
equilibrium and containment scores separately provide action priors for tree
selection.

## Source references

- [Reward implementation](../packages/stacking-common/src/stacking_common/environment/reward.py)
- [Voxel overlap](../packages/stacking-common/src/stacking_common/environment/overlap.py)
- [Surface relief](../packages/stacking-common/src/stacking_common/environment/surface.py)
- [Disturbance trials](../packages/stacking-common/src/stacking_common/environment/robustness.py)
- [Pose residual normalization](../packages/stacking-common/src/stacking_common/actions/core_solver.py)
- [Standard planar scoring](../packages/stacking-common/src/stacking_common/actions/_scored_floor.py)
- [Bottom-surface support kernels](../packages/stacking-common/src/stacking_common/actions/_support_maps.py)
- [Layer policy](../packages/stacking-common/src/stacking_common/actions/floor.py)
- [Composite sampling](../packages/stacking-common/src/stacking_common/actions/composite_sampling.py)
- [Root CEM](../packages/stacking-common/src/stacking_common/actions/cem.py)
- [Action priors](../packages/stacking-common/src/stacking_common/search/policies.py)
- [Search composition and guard](../packages/stacking-common/src/stacking_common/workflows/search.py)
- [Excavator reward configuration](../apps/excavator/config/environment.yml)
- [Excavator sampling configuration](../apps/excavator/config/action_generation.yml)
- [Excavator MCTS configuration](../apps/excavator/config/mcts_cuda.yml)
- [Tabletop reward configuration](../apps/tabletop/config/environment.yml)
- [Tabletop sampling configuration](../apps/tabletop/config/action_generation.yml)
