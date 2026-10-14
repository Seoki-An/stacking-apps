# Initial placement pose sampling for automated stone stacking

Implementation summary for a conference paper, inspected on **2026-10-06**.
Tabletop's scored XY sampling was aligned with excavator on **2026-10-07**.

This document describes the current working-tree implementation in
`stacking-apps`, with excavator as the primary platform and tabletop as a
comparison. Values below are the current repository defaults. A recorded
experiment may use a different profile, command-line overrides, or saved
configuration; its effective settings must be reported separately.

## Working version and score inspection

The planner uses the pre-region strategy from `824d204`, with consolidated
floor configuration, renamed planar bias settings, and five score terms.
Open-area and frontier bonuses have been removed; connectivity is a hard sampling mask. The regional experiment
is retained in the existing Git stash; files
that were untracked when stashing are backed up under
`.backups/region-sampling-20261011/` with their paths and the stash hash.

In the excavator operations console, select a candidate in **Search Inspector**
and click **XY score maps…**. The view compares scene height with the restored
sampler's total score, individual terms, and weighted contributions. It also
shows the target base, initial XY, and projected stone footprint, and can export
a PNG. Computation occurs only when opening the view, outside the MCTS loop.

The view uses the run's recorded configuration and pre-search scene. It fixes
the candidate's initial orientation and recomputes height normalization over
that orientation's eligible cells with normal floor gates. Thus it is a
diagnostic reconstruction, not an exact replay of multi-orientation scores or
relaxed retry sampling. Runs recorded with the regional scorer are explicitly
unsupported by this restored score viewer. Runs with nonzero removed `open_area`
or `frontier` weights are also rejected; zero historical weights are discarded.
Historical connectivity settings are accepted only when they match the fixed
one-cell rule without an active delay in the displayed state. Runs whose
recorded layer cap is reached are rejected by the viewer, since their capped
floor context cannot be reconstructed with the current policy.

## 1. Purpose and algorithm structure

Initial placement poses provide starting points for the native placement
optimizer. The generator combines a finite orientation bank with scene-aware
XY proposals, computes a conservative starting height, and refines each pose
with DSF contact and equilibrium optimization. Accepted actions subsequently
enter MCTS evaluation and simulation.

The sampled variables are an object identity, a planar position, and an
orientation selected from a structured bank. Initial Z is computed from
geometry. At the root of the search tree, an additional population refinement
stage perturbs XY and yaw around promising initial poses.

```mermaid
flowchart LR
    A[Available stones and placed scene] --> B[Object orientation bank]
    B --> C[Grid proposals and scene scores]
    C --> D[Object allocation and proposal selection]
    D --> E[Geometric initial Z]
    E --> F[Native pose refinement and filtering]
    F --> G[Root population refinement]
    G --> H[MCTS evaluation and simulation]
    F --> H
```

The root refinement branch applies at the root; ordinary deeper-node expansion
uses the platform generator directly. Simulation-based stability and execution
feasibility are subsequent stages, so a selected initial pose is a hypothesis,
not a certified placement.

## 2. Notation and geometry

Let the current stacking state be $s$, with available objects
$\mathcal I(s)$ and placed objects $\mathcal J(s)$. For candidate object
$i$, an initial pose is

$$
T_i^0=(R_i^0,t_i^0),\qquad t_i^0=(x_i,y_i,z_i^0).
$$

The Python representation is
`[x, y, z, qx, qy, qz, qw]`, using scalar-last unit quaternions. Placement search
uses the stack frame; platform composition supplies the transform to the robot
world frame for motion planning.

Several geometries participate in initialization:

| Geometry | Role in initial pose generation |
|---|---|
| Convex component nodes and faces from `model_<id>.obj` | Orientation construction, height-map rendering, component-volume queries, and tabletop initial bounds |
| Global convex hull of the component nodes | Excavator PCA and orientation-dependent bottom kernels |
| Reconstructed `model_<id>_mesh.ply` vertices | Excavator initial-height bounding boxes, with component vertices as fallback |
| Smooth DSF components | Native contact queries and pose refinement |

The hulls used by proposal heuristics differ from both the reconstructed
surface and the smooth DSF envelope. In particular, a global hull can bridge
concavities between components.

Both platforms render a **128 × 128** top-surface height map from the placed
component meshes. The bounds are a square centered on the target XY origin,
with side length

$$
L=\max(W,\ell)+m,
$$

where $W$ and $\ell$ are target width and length, and $m$ is the
configured height-map margin. Both XY proposal grids have **32 × 32** samples
spanning these same bounds, including the endpoints. Their spacing is
$L/31$, followed by a target-region mask. Thus 32 × 32 denotes the complete
observation domain, not 32 × 32 valid anchors inside the target.

| Quantity | Excavator | Tabletop |
|---|---:|---:|
| Target width × length × height | 2 × 4 × 2.5 m | 0.08 × 0.17 × 0.15 m |
| Height-map margin parameter | 1 m | 0.02 m |
| Square observation span $L$ | 5 m | 0.19 m |
| XY proposal spacing $L/31$ | Approximately 0.1613 m | Approximately 0.00613 m |
| Target inward taper | 5° on all four sides | 10° on both length sides; 0° on width sides |

## 3. Excavator orientation hypotheses

Configure the yaw bank with `orientation.yaw_angles_degrees` on both platforms.
Excavator's roll bank is `orientation.roll_angles_degrees`. These rotations are
combined before bottom-kernel construction and score-map computation. The
resolved sampler still stores yaw angles in `PlanarSamplingConfig`; moving the
YAML key does not change the generated orientations or recorded runtime profile.

The excavator profile sets `pose_from_scan: false`. Its base orientation is
constructed from the global hull vertices of the convex-component model.

For centered vertices $v_k-\bar v$, the implementation forms the covariance
matrix

$$
C=\frac{1}{N}\sum_{k=1}^{N}(v_k-\bar v)(v_k-\bar v)^\mathsf T.
$$

The eigenvector with the largest eigenvalue defines the longest principal
axis; the smallest defines the shortest. The longest axis becomes X and the
shortest becomes Z. Axis signs are resolved by making the largest-magnitude
coordinate positive, and cross products complete a right-handed basis. If
$P$ has these axes as columns, the canonical body-to-stack rotation is
$R_{\mathrm{PCA}}=P^\mathsf T$.

Initial rotations are constructed as

$$
R_i^0=R_z(\psi)R_x(\phi)R_{\mathrm{PCA}},
\quad
\phi\in\{0,\pi\},
\quad
\psi\in\{0,\tfrac{\pi}{2},\pi,\tfrac{3\pi}{2}\}.
$$

This yields up to eight distinct rotations per object. Numerically equivalent
rotation matrices are deduplicated by the scored sampler. These hypotheses
favor placing the stone on one of two opposite faces associated with its
shortest principal axis. PCA is a geometric heuristic; the orientations are
not established stable equilibria before native refinement.

With `pose_from_scan: true`, excavator instead uses the supplied inventory
orientation as its base and applies the configured yaw angles. That alternative
also enables a scan-relative orientation filter. It is not the current default.

## 4. Excavator XY anchors and support estimates

### 4.1 Target mask and floor footprint

XY anchors are drawn from the regular proposal grid. For each initial rotation
$R$ (including roll and yaw), the sampler transforms the stone's hull vertices
and caches their XY minima $a$ and maxima $b$, relative to the stone origin.
With target bounds $L,U$, `target_anchor_inset_scale` $s$ and
`target_mask_margin` $m$, allowed anchor coordinates satisfy

$$
L + m - s a \;\leq\; (x,y) \;\leq\; U - m - s b.
$$

The current $s=0.6$ and $m=0$ permit initial overhang. At $s=1$, these bounds
contain the rotated XY bounding box in the nominal target rectangle. Keeping
both offsets handles geometry whose origin is not at its bounding-box center.
The target-boundary score uses the same offsets with the target taper evaluated
at the estimated contact height. Final containment is checked after pose solving.

For gap filling and connectivity filtering, both platforms use a rectangle
from $(x,y)+f a$ to $(x,y)+f b$, where $f$ is `floor.footprint_scale` (currently
0.9). The offsets are scaled about the stone origin. This is an axis-aligned
approximation of the rotated shape; the separate bottom-height support kernel
retains its more detailed orientation-dependent geometry.

Target eligibility, connectivity, and empty-cell counts are
computed per orientation. Geometry bounds are cached alongside bottom kernels
across MCTS states. Summed-area tables count rectangle coverage with four
lookups, avoiding an orientation-by-anchor-by-floor-cell mask allocation.
An aggregate score-map cell displays the best eligible orientation when one
exists; otherwise it retains a raw score for diagnosis and is not sampleable.
The GUI's fixed-orientation view uses these same rules. Historical runs are
recomputed with the current rotated-footprint policy, so they can differ from
the former asset-local rectangle approximation.

A local CPU check used the saved `scored_sampling.npz` placed-scene fixture,
two excavator stones, eight orientations, and 30 timed recomputations after
five warmups, retaining the geometry cache but invalidating the scene cache.
Median map times were 3.87 and 3.92 ms, versus 2.93 and 2.57 ms with commit
`b4ef8b3`. Candidate counts changed from 576/512 to 640/576 because eligibility
now follows rotation. These are scorer measurements, not full MCTS timings.

### 4.2 Orientation-dependent bottom kernel

For each rotation, the sampler rasterizes the lower surface of the global
convex hull onto a sparse set of grid offsets. Let $b_R(u)$ denote the
object's local bottom height at footprint offset $u$, and let
$h_s(a+u)$ be the scene height at XY anchor $a$ plus that offset.

The estimated vertical translation that clears the sampled scene is

$$
\widehat z(a,R)=\max_{u}\big[h_s(a+u)-b_R(u)\big].
$$

An offset is treated as supporting the object if

$$
\widehat z(a,R)+b_R(u)-h_s(a+u)\le\epsilon_c,
\qquad \epsilon_c=0.10\ \mathrm m.
$$

The support-likelihood score is the fraction of kernel offsets satisfying this
condition. The scorer also records the mean scene height over these contact
offsets. A kernel extending outside the height-map domain produces an invalid
support score and is omitted.

The term called candidate `height` in the excavator scorer is this mean contact
surface height. It differs from the translation estimate $\widehat z$, the
candidate's lowest point, and the later bounding-box initialization height.

## 5. Excavator proposal score and selection

### 5.1 Score terms

The excavator YAML keeps initial planar sampling settings together under
`planar` in [action_generation.yml](../apps/excavator/config/action_generation.yml):

| Configuration block | Terms | Consumer |
|---|---|---|
| `planar.score` | Five named weights: `height`, `support_likelihood`, `fill_area`, `target_boundary`, `planar_bias` | Excavator's `ScoredPlanarPoseSampler`, through `_scored_floor.score_xy_debug_map` |
| `planar.floor` | Coverage mode, fill ratio, grid size, bottom tolerance, footprint scale | Shared by the scored sampler and `ActiveFloorPolicy` |

Both platforms use one `planar.floor` block. Excavator's values are:

```yaml
floor:
  coverage_mode: projected_footprint
  bottom_tolerance: 0.2
  fill_ratio: 0.75
  grid_size: 16
  footprint_scale: 0.9
```

Tabletop explicitly configures `bottom_slice`, 0.02 m bottom tolerance, 0.90
fill ratio, and a 32-cell grid. Its search and planning floor checks read the
same YAML values. Sampling and solved-pose connectivity both require at least
one neighboring occupied cell; `minimum_contact_cells` is no longer configurable. Sampling connectivity starts whenever
the active floor contains occupied cells; the configurable `connected_after`
delay has been removed. An empty active floor allows a seed placement.
`footprint_scale` sizes the sampling rectangle.
The U-shape pocket filter and its settings have been removed; these footprint
and connectivity controls remain active.
`filtering.strict_floor_first` still controls the additional solved-pose check.

The old `floor_fill` block and duplicate `lower_floor_*` keys are rejected in
new profiles. Saved split configurations remain supported by the score viewer.
Resume compares recorded profiles, so use a new run for this schema change.

`planar.score` is the only excavator proposal-weight source. The shared
`ActiveFloorPolicy` constructor has no score weights. Tabletop now uses the
same named `planar.score` block in its own profile. The old six-term weights
on `TabletopPlacementSafety` were removed with the obsolete deterministic
sampler after alignment.

Before consolidation, excavator had root-level `floor` and `scored_sampling`
blocks. Its six-entry `floor.score_weights` list was unused by the excavator
sampler; the seven named `scored_sampling.score` weights were effective.
The old list is removed, floor geometry moved to `planar.floor`, and the
contents of `scored_sampling` moved under `planar`. Obsolete YAML blocks are
rejected so that edits cannot silently target inactive weights. Effective
weights and sampling behavior are preserved. Saved planning contexts still
record runtime `floor_settings` and `sampling_settings` separately.

In the historical `excavator_plan8_gpu.stack` and `excavator_plan9_gpu.stack`,
the only recorded planning-context difference is the unused floor-weight
list: respectively $[-3,2,0,2,0,1.5]$ and $[-3,2,1,2,1,1.5]$.
Their active seven named weights are identical. Those files therefore do
not record an excavator proposal-score ablation.

Each retained anchor/orientation pair receives the following score. Current
weights on both platforms are $(-3,2,2,1.5,0)$ for $(H,Q,F,B,D)$.

$$
S=w_HH+w_QQ+w_FF+w_BB+w_DD.
$$

| Term | Current definition | Intended geometric bias |
|---|---|---|
| $H$ | Mean contact height normalized over scored anchor/orientation candidates before connectivity masking | Lower supporting surfaces |
| $Q$ | Fraction of bottom-kernel samples within the contact tolerance | Broader approximate support |
| $F$ | Unoccupied footprint-cell count normalized over open anchors before connectivity filtering | Covering a larger unfilled area |
| $B$ | Maximum absolute normalized XY coordinate relative to the height-dependent target bounds | Positions nearer a target boundary |
| $D$ | Distance to a reference point, divided by target XY diagonal and clipped to one | Greater distance from that reference |

The following equations give the exact geometric definitions used by the
current scorer. For a fixed object, let $a=(a_x,a_y)$ be an XY anchor,
$R$ its candidate rotation, $\mathcal K_R$ its sparse bottom-kernel offsets,
and $\mathcal C$ the retained anchor/orientation pairs with finite support
scores. Let $\mathbf 1[\cdot]$ denote an indicator and $|\cdot|$ a set's
cardinality. All counts refer to sampled grid cells.

#### Support likelihood $Q$

Let $h_s$ be the scene height map after maximum-based resizing to the score
grid and clipping upward to the ground height. With the bottom kernel
$b_R(u)$ from Section 4.2, define

$$
\widehat z(a,R)=\max_{u\in\mathcal K_R}
\left[h_s(a+u)-b_R(u)\right],
\qquad
g_u(a,R)=\widehat z(a,R)+b_R(u)-h_s(a+u).
$$

The estimated contact offsets and support score are

$$
\mathcal K_c(a,R)=
\left\{u\in\mathcal K_R:g_u(a,R)\le\epsilon_c\right\},
\qquad
Q(a,R)=\frac{|\mathcal K_c(a,R)|}{|\mathcal K_R|}
=\frac{1}{|\mathcal K_R|}\sum_{u\in\mathcal K_R}
\mathbf 1[g_u(a,R)\le\epsilon_c],
$$

where $\epsilon_c=0.10$ m. A nonempty kernel with complete finite scene
coverage has at least one offset at zero estimated gap. An empty kernel or
incomplete coverage gives an invalid score, and that candidate is omitted.

#### Normalized contact height $H$

The raw height is the mean scene height over estimated contact offsets:

$$
h_c(a,R)=\frac{1}{|\mathcal K_c(a,R)|}
\sum_{u\in\mathcal K_c(a,R)}h_s(a+u).
$$

Define extrema over the scored pairs for this object in the current state,
after target/open-space/support validity checks and before connectivity masking:

$$
h_{c,\min}=\min_{(a,R)\in\mathcal C}h_c(a,R),
\qquad
h_{c,\max}=\max_{(a,R)\in\mathcal C}h_c(a,R).
$$

The height term is

$$
H(a,R)=\frac{h_c(a,R)-h_{c,\min}}
{\max(h_{c,\max}-h_{c,\min},10^{-6}\ \mathrm m)}.
$$

The negative height coefficient therefore favors low contact surfaces. The
code has a fallback to mean footprint height if the reported contact height
is nonfinite; the equations above describe the finite-support path.

#### Floor footprint and unfilled cells

Let $\mathcal G_f$ be the active-floor sampling grid, $\Omega_s$ its occupied
cells, and $q_R^-,q_R^+$ the XY minima and maxima of the rotated vertices.
At anchor $a$, the scorer's rectangular footprint is

$$
\mathcal A(a,R)=\left\{p\in\mathcal G_f:
a+f q_R^-\le p\le a+f q_R^+\right\}.
$$

The configured footprint scale is $f=0.9$. Define

$$
\mathcal U(a,R)=\mathcal A(a,R)\setminus\Omega_s,
\qquad n_A(a,R)=|\mathcal A(a,R)|,
\qquad n_U(a,R)=|\mathcal U(a,R)|.
$$

In the following expressions, $R$ is implicit in $n_A,n_U$ and $\mathcal A$.
When the active floor contains occupied cells and floor filling is not relaxed,
anchor/orientation pairs with $n_U=0$ are removed at this stage. This is an
admission rule, not an open-area score.

#### Normalized fill area $F$

Let $\mathcal A_{\mathrm{open}}$ contain base-mask-valid anchor/orientation pairs with
$n_A>0$ and $n_U>0$, before the later connectivity filter. The normalization is

$$
n_{U,\max}=\max\left(1,
\max_{(a,R)\in\mathcal A_{\mathrm{open}}}n_U(a,R)\right),
\qquad
F(a,R)=\frac{n_U(a,R)}{n_{U,\max}}.
$$

If that set is empty, the code keeps the normalization denominator at one.
$F$ compares the unoccupied cell count with other anchor/orientation pairs for the same object.
The anchor attaining $n_{U,\max}$ can be removed by a later filter.

#### Connectivity mask

Use integer row/column coordinates on $\mathcal G_f$ to define the
eight-neighbor expansion

$$
\mathcal N_8(\Omega_s)=
\left\{p\in\mathcal G_f:\exists q\in\Omega_s,
\quad p-q\in\{-1,0,1\}^2\setminus\{(0,0)\}\right\}.
$$

The expansion uses eight shifts and excludes the zero shift. It can still
include occupied cells if they neighbor other occupied cells. Connectivity
requires at least one footprint cell in this expansion;
there is no weighted frontier bonus.

Connectivity is applied as a hard mask after raw score computation:

$$
M_C(a,R)=\mathbf 1[|\mathcal A(a,R)\cap\mathcal N_8(\Omega_s)|\ge 1].
$$

On an empty active floor, $M_C=1$ for target-valid anchors. The score-map API
returns raw `scores`, a separate `connectivity_mask`, and the final
`candidate_mask`. Only admitted candidates enter elite or Boltzmann
selection. Connectivity changes admission, not raw score values; the final
score normalization uses the admitted pool. If that pool is
empty, sampling returns no proposals. Floor-fill retries retain this mask.
The estimate uses coarse rectangular footprints, so later pose optimization
and physical validation still determine actual contact.

If there is no occupied active-floor mask, the fill-area score defaults to
**$F=0$**. Floor-fill retries retain this score; only the hard requirement
$n_U>0$ is relaxed.

#### Target-boundary proximity $B$

Let the target origin be $o=(o_x,o_y,o_z)$, dimensions $(W,\ell,h_t)$,
and inward side angles $\theta_x^-,\theta_x^+,\theta_y^-,\theta_y^+$.
Let $\operatorname{clip}(v,l,u)=\min(\max(v,l),u)$, and define

$$
\zeta(a,R)=\operatorname{clip}(h_c(a,R)-o_z,0,h_t),
\qquad
\delta^- = m_t-s q_R^-,\quad \delta^+ = m_t+s q_R^+,
$$

where the current target-mask margin is $m_t=0$ and inset scale is $s=0.6$. For the scorer, set
$\alpha_d^\pm=\tan(\max(\theta_d^\pm,0))$. Negative taper angles are
therefore clamped to zero in this heuristic. The bounds are

$$
\begin{aligned}
l_x&=o_x-\frac W2+\zeta\alpha_x^-+\delta_x^-,&
u_x&=o_x+\frac W2-\zeta\alpha_x^+-\delta_x^+,\\
l_y&=o_y-\frac\ell2+\zeta\alpha_y^-+\delta_y^-,&
u_y&=o_y+\frac\ell2-\zeta\alpha_y^+-\delta_y^+.
\end{aligned}
$$

Set $l=(l_x,l_y)$, $u=(u_x,u_y)$, center $c=(l+u)/2$, and half-span
$r=(u-l)/2$. With $\oslash$ denoting componentwise division,

$$
B(a,R)=
\begin{cases}
\operatorname{clip}\left(\left\|(a-c)\oslash r\right\|_\infty,0,1\right),
& r_x>0,\ r_y>0,\ l\le a\le u,\\
0,&\text{otherwise}.
\end{cases}
$$

The vector inequalities hold componentwise. $B=0$ at the center and $B=1$
on a valid boundary. Anchors outside these height-dependent bounds receive
zero boundary bonus; this expression does not remove them from the initial
base-rectangle mask.

#### Reference-distance score $D$

For the currently configured XY distance mode and reference $a_e$, the score
is

$$
D(a)=\min\left(\frac{\|a-a_e\|_2}
{\max(\sqrt{W^2+\ell^2},10^{-6}\ \mathrm m)},1\right).
$$

The implemented single-axis alternatives are

$$
D_x(a)=\min\left(\frac{|a_x-a_{e,x}|}
{\max(x_{\max}-x_{\min},10^{-6}\ \mathrm m)},1\right),
\qquad
D_y(a)=\min\left(\frac{|a_y-a_{e,y}|}
{\max(y_{\max}-y_{\min},10^{-6}\ \mathrm m)},1\right),
$$

where the axis spans are the observation bounds. The XY version normalizes
by the nominal target diagonal, rather than the observation diagonal.

Configure this term with `planar.score.planar_bias` (signed weight),
`planar_bias_axis` (`x`, `y`, or `xy`), and `planar_bias_xy` (reference XY in metres).
The current weight is zero on both platforms, so the term contributes nothing.
Older saved contexts using `excavator_distance`, `excavator_distance_axis`, and
`excavator_xy` remain readable; score maps and the GUI expose `planar_bias`.

With the current `planar_bias_xy: null`, the distance reference is **[0, 0]** in
the search frame. A positive distance weight rewards farther positions. The
paper should describe this direction accurately rather than calling the term
a preference for shorter robot travel. Likewise, boundary proximity rewards
edge proximity rather than a center-seeking preference.

On an empty active floor, the fill-area term is zero and the remaining terms
drive selection. The score is a proposal
heuristic, distinct from both MCTS return and the native pose objective.

### 5.2 Continuous integral formulation for the paper

The spatial averages and footprint counts above can be written as integrals.
The following equations are **continuous counterparts**; the implementation
evaluates sampled masks, sums, and maxima on finite grids. Use planar area
measure $d^2u=du_x\,du_y$, rather than area measured along the sloping stone
surface: the code gives equal weight to XY samples.

#### Writing convolution explicitly as an integral

For a planar kernel $k$ and a field $f$, convolution means

$$
(k*f)(a)=\int_{\mathbb R^2}k(a-p)f(p)\,d^2p.
$$

The masked mean filter in `_support_maps.py` uses a square box kernel. Let
$M(p)$ indicate valid, finite height samples, extend it by zero outside the
observation domain, and define $h_M(p)=h_s(p)$ where $M(p)=1$ and zero
elsewhere. Let $k_5$ indicate a centered square with width five grid spacings
along each axis. Its continuous counterpart is

$$
\widetilde h_s(a)=
\frac{\displaystyle\int_{\mathbb R^2}
k_5(a-p)h_M(p)\,d^2p}
{\displaystyle\int_{\mathbb R^2}
k_5(a-p)M(p)\,d^2p},
\qquad\text{if the denominator is positive}.
$$

If there are no included samples, the code retains the original height at
$a$. This denominator also prevents samples outside the valid domain from
diluting the mean. With piecewise constant grid cells, the integral at a grid
center reproduces the discrete $5\times5$ masked mean. Excavator's main
support calculation uses the **unsmoothed** map $h_s$; the smoothed map serves
the fallback footprint-height calculation.

For an arbitrary footprint $\mathcal P$, its translated overlap with $f$
has the explicit integral

$$
\int_{\mathcal P}f(a+u)\,d^2u
=\int_{\mathbb R^2}\mathbf 1_{\mathcal P}(p-a)f(p)\,d^2p
=(\mathbf 1_{-\mathcal P}*f)(a),
$$

where $-\mathcal P=\{-u:u\in\mathcal P\}$. Thus sliding-footprint
averaging is a correlation, or equivalently a convolution with a reflected
footprint. For the centered rectangular excavator footprint,
$-\mathcal P=\mathcal P$.

#### Support coverage and contact height

Let $\mathcal P_R$ be the continuous XY projection corresponding to the
bottom kernel, and $b_R(u)$ its lower surface. The support-height estimate
and gap field are

$$
\widehat z(a,R)=\sup_{u\in\mathcal P_R}
\left[h_s(a+u)-b_R(u)\right],
\qquad
g(a,R,u)=\widehat z(a,R)+b_R(u)-h_s(a+u).
$$

Define the contact indicator
$c_{a,R}(u)=\mathbf 1[g(a,R,u)\le\epsilon_c]$. Then

$$
Q(a,R)=
\frac{\displaystyle\int_{\mathcal P_R}c_{a,R}(u)\,d^2u}
{\displaystyle\int_{\mathcal P_R}1\,d^2u},
\qquad
h_c(a,R)=
\frac{\displaystyle\int_{\mathcal P_R}
h_s(a+u)c_{a,R}(u)\,d^2u}
{\displaystyle\int_{\mathcal P_R}c_{a,R}(u)\,d^2u}.
$$

These ratios require positive footprint and contact areas, respectively.
The discrete implementation uses the sampled contact set, which can contain
a single sample even if its continuous counterpart has zero area. Normalize
$h_c$ over the retained candidate bank using the same $H$ equation in
Section 5.1. The support supremum and gap threshold are nonlinear operations;
the complete support score cannot be replaced by an ordinary linear
convolution.

#### Fill area and connectivity

Let $\mathcal T_{xy}$ be the nominal target rectangle,
$\chi_s(p)=\mathbf 1[p\in\Omega_s]$ the active-floor occupancy field,
and

$$
\rho_i(u)=\mathbf 1[|u_x|\le0.45e_x,\ |u_y|\le0.45e_y].
$$

The continuous footprint area and unfilled area are

$$
A_i(a)=\int_{\mathcal T_{xy}}\rho_i(p-a)\,d^2p,
\qquad
U_i(a)=\int_{\mathcal T_{xy}}
\rho_i(p-a)\left[1-\chi_s(p)\right]\,d^2p.
$$

To retain the implemented eight-neighbor construction, let
$\Delta_x,\Delta_y$ be the occupancy-grid spacings and define

$$
\mathcal E_8=
\left\{(j\Delta_x,k\Delta_y):j,k\in\{-1,0,1\},\ (j,k)\ne(0,0)\right\},
\qquad
\nu_s(p)=\mathbf 1[\exists\delta\in\mathcal E_8:
\chi_s(p+\delta)=1].
$$

Occupancy is zero outside its grid domain. This is the binary union of eight
shifted occupancy masks; it excludes the zero shift. Define the neighboring
footprint area

$$
L_i(a)=\int_{\mathcal T_{xy}}\rho_i(p-a)\nu_s(p)\,d^2p.
$$

The fill-area score becomes

$$
F(a)=\frac{U_i(a)}
{\max\left(\Delta A_f,\max_{a'\in\mathcal A_{\mathrm{open}}}U_i(a')\right)}.
$$

Here $\Delta A_f=\Delta_x\Delta_y$ is the floor grid's uniform area weight;
it expresses the discrete denominator safeguard of one sample in area units.
An empty normalization set uses $\Delta A_f$. Empty-floor scoring uses $F=0$;
relaxed scoring retains $F$. Connectivity is a separate mask $L_i(a)\ge\Delta A_f$ when the
floor is occupied, including relaxed retries. Its integral includes occupied
neighboring cells as well as unfilled ones.

For uniform samples, replacing each integral by
$\Delta A\sum_p$ recovers the count ratios in Section 5.1; the common area
weight cancels. The grids include endpoints and give every sample equal
weight; their sums are not trapezoidal integration with reduced boundary
weights. Finite masks, sparse kernel construction, and grid boundaries still
determine the numerical approximation. $B$ and $D$ remain pointwise
geometric expressions. The combined score remains
$S=w_HH+w_QQ+w_FF+w_BB+w_DD$.

### 5.3 Elite and Boltzmann draws

Within a candidate pool, weighted draws use

$$
p_j=\frac{\exp(S_j/\tau)}{\sum_{k\in\mathcal P}\exp(S_k/\tau)},
\qquad \tau=0.3,
$$

over the currently unselected members of that pool. Numerical evaluation
subtracts the maximum logit before exponentiation. The probability floor is
zero.

Within each eligible priority pool, selection now has two stages:

1. Take the highest-scoring $\min(\lceil f k\rceil,n)$ candidates as elites,
   where $f=0.25$, $k$ is the requested count and $n$ is the available count.
2. Fill the remaining slots with Boltzmann draws without replacement from all
   unselected candidates in that pool.

For positive requests, the result contains $\min(k,n)$ distinct candidates.
Elite counts retain ceiling rounding: a one-candidate request with nonzero
elite fraction chooses the highest-scoring candidate. Lower-floor priority
still divides candidates into lower and higher pools before these stages.

Y-band, gap-component, and height/quadrant representative draws have been
removed. Gap labeling, component-center calculations, and per-candidate gap
metadata are no longer computed. Fill-area scores still count empty floor
cells, and connectivity still uses the occupied-cell neighborhood mask.
Score-map values and eligibility masks are unchanged by this removal, while
seeded proposal sequences change on both platforms.

There is no minimum-distance thinning between XY proposals. Root population
deduplication later applies its own thresholds. Object allocation and subsequent
generator-level selection still contribute to the overall proposal distribution;
it is not a single global softmax across all objects and floor priorities.

### Empty-scene corner seeding

When `state.placed_object_indices` is empty, both platforms restrict fresh XY
proposals to the four feasible target corners. This tests the actual scene
state, so a newly active upper floor does **not** restart corner seeding.

For each initial orientation, rotate the mesh hull and compute its signed XY
minimum/maximum offsets. Subtract these offsets from the target bounds to obtain
feasible **center** coordinates. Corner seeds use at least the full footprint
(`max(target_anchor_inset_scale, 1)`); ordinary sampling may still permit the
configured overhang. Existing target margins and all score-map eligibility
checks still apply. Among eligible grid cells, take those within one grid
interval in each axis of a feasible corner. No extra score weight or neighborhood
radius is configured. For narrow bounds, each cell belongs to its nearest corner
only, avoiding duplicate proposals.

Shuffle the available corners and draw one seed per corner while budget remains,
using the existing score selector within each corner. Fill any remaining slots
with elite/Boltzmann draws from unselected corner seeds. Small requests therefore
have an opportunity to visit every corner across draws; four-or-more proposals
cover all available corners. Generator allocation retains that shuffled prefix.
Scores themselves remain unchanged, and CEM/pose solving may move proposals away
from a corner. MCTS uses its normal reward and validation ranking. This is an
initial-position preference, not a constraint on the final settled pose.

If no eligible corner cells exist, immediately use ordinary scored sampling.
If corner proposals produce no valid root selection after generation/physical
validation, the existing final retry stage uses ordinary sampling within the
original deadline and retry budget. This retry requires a final-validation
request and available time. It preserves floor priorities and filters, physical
checks, and reward weights. The temporary sampler override is restored on exit.
Once any stone has been placed, descendants use ordinary sampling and the
existing connectivity filter. No region cursor is introduced.

Proposal diagnostics record `sampling_policy: corner_seed` and `seed_corner`
(0: lower-X/lower-Y, then counterclockwise), or `corner_seed_fallback: true` for
ordinary empty-scene proposals. Score-map debug data retains the complete score
map and exposes the seed pool as `corner_candidates`. For artifact compatibility,
root retry statistics retain the `floor_fill_fallback` field, with
`policy: corner_seed` identifying this new retry; ordinary floor retries have
`policy: floor_fill`.

### 5.4 Allocation across objects

Both platforms set `stone_sampling.volume_exponent: 1.0` in
`action_generation.yml`. Each fresh proposal slot is assigned to an unplaced,
non-banned stone with probability

$$
P(i)=\frac{V_i^\alpha}{\sum_{j\in\mathcal E} V_j^\alpha},
\qquad \alpha=\texttt{volume\_exponent},
$$

where $V_i$ is the inventory geometry volume and $\mathcal E$ excludes stones
that have reached the per-request proposal cap (currently 32). Slots are drawn
with replacement until that cap; probabilities are renormalized when a stone
reaches it. Positive-volume smaller stones remain eligible. Volumes and weights
are cached when constructing the generator, outside the MCTS iteration loop.
This biases proposal opportunities, not MCTS reward or the within-stone XY score.
It applies to fresh root/CEM proposals, descendants, and relaxed retries; CEM's
separate retained-candidate per-object cap still applies. Mutation parent
selection remains governed by CEM scores.

Setting the exponent to zero (also the default for older profiles without this
block) restores the previous round-robin allocation: shuffled objects without
lower-floor priority, rotating the starting object for small floor-first requests.
For each object's allocated budget, the generator takes a prefix of its sampled
list when lower-floor priority is active or corner seeding applies. Otherwise,
it randomly selects from that list. The exponent is recorded in the resolved
generator profile for run-context compatibility checks.

The sampler draws a fresh population on each call using persistent private
Python and NumPy random-number generators. Retry offsets do not advance a
fixed deterministic list as they did for the previous tabletop generator.
Both current platforms use this fresh-population allocation procedure.

Tabletop's CPU and CUDA MCTS profiles now use lookahead from the first placement
and mixed final selection with a required, validated two-step continuation,
matching excavator's selection policy. This enables the descendant-only floor
relaxation described below. Both retain at most two CEM candidates per object
and draw 50% fresh proposals in later CEM rounds. Tabletop retains its own search
budgets, scan orientations, physical tolerances, and reward weights. Its pose
solver now uses `k_lower: {anti_ramp: 10.0}`; structural simulation uses
`limit_surface_6d` with `patch_eps: 0.00003`. These physical changes require
tabletop saved-scene validation before claiming improved placement performance.

## 6. Active-floor policy

Excavator's main composition shares one `ActiveFloorPolicy` between proposal
sampling and filtering. Its current settings are:

| Setting | Excavator | Tabletop |
|---|---:|---:|
| Occupancy representation | Projected union of convex-component footprints | Component-volume intersections with a bottom slab |
| Occupancy grid | 16 × 16 cell-center samples | 32 × 32 samples including endpoints |
| Floor completion fraction | 0.75 | 0.90 |
| Bottom-height tolerance | 0.20 m | 0.02 m |
| Connectivity rule (fixed) | At least one neighboring occupied cell | Same |
| Strict action-build floor filter | Enabled | Enabled |

Starting at the target base, the policy collects stones whose bottom is within
the current layer tolerance. If their union covers the required fraction, it
advances the support height to the highest top among those stones and repeats
with remaining stones. The first insufficiently filled layer is active.
There is no configurable layer cap: every completed layer consumes at least
one stone, and detection stops when no stones remain or no stones belong to
the current layer. Removing `max_layers` also avoids misclassifying occupied
layers above the old cutoff as empty.

For either platform's `lower_floor_first` sampling, candidates whose estimated
contact height is at most active support height plus the platform's bottom
tolerance are selected before higher candidates: 0.20 m for excavator and
0.02 m for tabletop. The strict filter separately checks the bottom
height and connectivity of the **solved** pose. These checks act on different
quantities and should be described separately.

After a placed footprint exists, either platform's strict scored proposal
construction requires some open footprint area and at least one cell touching
dilated occupancy. No U-shaped-pocket rejection is applied.

Before rejecting a physically validated root for a missing required continuation,
the search tries to repair its descendant path with relaxed floor admission.
This happens once per root candidate, after ordinary path repair, within the
existing deadline and bounded repair budget. The root pose and its physical
validation are preserved; existing descendants and physical failures remain
attached. Exhausted descendant generation is reopened for the changed admission
rule. Newly generated descendants still require physical validation. Diagnostic
fields `floor_fill_continuation_attempted` and `floor_fill_continuation_recovered`
record this repair on the root candidate. When a repaired branch is promoted for
the next placement, its relaxed immediate continuations are rebuilt under the
strict policy.

If root selection still fails with a strict floor filter present, the search
tries a new root under relaxed floor-fill admission. Both this fallback and
descendant repair retain the fill-area score and any requested
`lower_floor_first` sampling priority. Native pose feasibility and subsequent
physical evaluation still apply, and sampled seeds still obey the connectivity
mask. Thus the
complete search policy permits an explicit floor-fill
fallback, rather than guaranteeing that every selected action satisfies the
strict lower-floor preference.

The current excavator retry limit allows an initial relaxed root pool and up
to three further retry pools, subject to the original deadline and pool
availability. Relaxation is conditional on a final-validation request and
available time.

The excavator completion threshold is `planar.floor.fill_ratio: 0.75`.
The old unused 0.6 fallback threshold has been removed from the profile.

## 7. Tabletop proposal strategy

### 7.1 Comparison of current XY sampling defaults

Both platforms enumerate a regular **32 × 32** anchor grid from a rendered
**128 × 128** scene height map. Neither platform starts by drawing continuous
uniform XY coordinates. Their initial grid proposals differ as follows:

| Step | Tabletop | Excavator |
|---|---|---|
| Sampler | `ScoredPlanarPoseSampler` | `ScoredPlanarPoseSampler` |
| Scene-map reduction | Block maxima, with a separate smoothed fallback map | Same |
| Default orientation bank | Scan-aligned orientation with retained tilt; two yaws, 0°/180° | PCA orientation; two rolls and four yaws, up to eight distinct rotations |
| Support geometry | Orientation-dependent global convex-hull bottom kernel | Same |
| Floor-filling geometry | Asset-local axis-aligned rectangular footprint on the active-floor grid | Same |
| Support contact tolerance | 0.02 m | 0.10 m |
| Active-floor occupancy | Bottom-slab volume intersections, 32 × 32; advance at 90% | Projected component footprints, 16 × 16; advance at 75% |
| Weighted score | $-3H+2Q+2F+1.5B$ | Same |
| Boundary feature | Bounds tapered according to estimated contact height | Same |
| Selection | Highest-scoring elites followed by seeded Boltzmann draws without replacement | Same |
| XY separation | No sampler distance thinning | Same |
| Repeated requests | New draws advance persistent private RNGs | Same |

Both compositions now use the same score definitions and selection code.
Their different geometry, orientation banks, physical tolerances, and floor
contexts still produce different candidate banks and rankings. Tabletop's
previous six-term sampler and its different feature definitions are recorded
in Section 7.3 for historical comparisons.

Both platforms enable `strict_floor_first: true` and prefer the lower
contact-height pool through `lower_floor_first`. Both apply the strict
solved-pose floor filter using their own physical floor settings.
The conditional relaxed root fallback is described in Section 6.

The shared strategy subsequently initializes Z and invokes native placement
optimization, which can move the initial XY coordinates.
Root population refinement can also create continuous XY perturbations.
The alignment changes tabletop proposal behavior; it does not demonstrate
improved physical placement or end-to-end search quality.

### 7.2 Current tabletop composition and configuration

Tabletop currently enables `pose_from_scan`. It finds the object's longest
local axis, transforms that axis with the measured orientation, and applies a
world-Z correction that aligns its projected heading with X modulo 180°. The
measured tilt is retained. The yaw bank then adds **0° and 180°**.

Both platforms inherit `StackingSearchApplication.build_action_generator`,
which constructs `ScoredPlanarPoseSampler` and
`ScoredPlacementActionGenerator`. Excavator overrides only its PLY-based
height initializer; tabletop retains `MeshHeightInitializer` with 0.015 m
clearance and 0.012 m padding.

Tabletop's [action_generation.yml](../apps/tabletop/config/action_generation.yml)
explicitly resolves temperature 0.3, elite fraction 0.25, the same five
dimensionless weights as excavator, and its own 0.02 m contact tolerance.
`planar_bias` measures normalized distance from the configured reference,
defaulting to the search-frame origin when `planar_bias_xy` is null. It does
not use robot geometry on either platform.

`TabletopPlacementSafety` continues to supply its 32 × 32 bottom-slab floor
context, 90% completion threshold, and 0.02 m bottom tolerance. Initial scan
orientation, solved-orientation filtering, target-containment checks, and
solver acceptance limits remain platform-owned.

Every effective sampler setting is recorded as
`action_generation_profile.sampling_settings` in tabletop planning contexts.
Together with the changed minimum-separation value, this makes the existing
resume validation reject contexts recorded with the previous sampler.
Historical artifacts remain readable; this change does not rewrite them.

### 7.3 Historical tabletop sampler before 2026-10-07

The rest of this section describes the former `HeightMapPlanarPoseSampler`
and the tabletop composition used before alignment. That implementation has
been removed from the live checkout. Its equations and
selection procedure are historical, not the current tabletop defaults.

The sampler enumerates valid grid anchors for each orientation. With the
then-configured target inset, the center bounds use 0.6 times the asset-local
half-extent. Full rotated bounding-box fitting is used only when that inset
option is absent.

Both root and deeper priorities are configured as `lower_floor_first`.
Tabletop therefore constructs orientation-dependent footprint kernels from
the union of component volumes. It computes support height, support
likelihood, unfilled area, frontier, and target-boundary features, with weights

$$
S_{\mathrm{tabletop}}=-3H+2Q+O+3F+G+1.5B.
$$

The letters have analogous roles, but tabletop computes them using the
following orientation-dependent geometry. Let $\mathcal K_R^t(a)$ be the
kernel offsets that remain inside the observation grid at anchor $a$.
For each offset, let $b_R(u)$ and $t_R(u)$ be the lower and upper heights
from the component union, and let $b_{\min}$ be the object's lowest rotated
vertex. Define

$$
\bar h_s(a+u)=\max(h_s(a+u),z_f),
\qquad
\widehat z_t(a,R)=\max_{u\in\mathcal K_R^t(a)}
\left[\bar h_s(a+u)-b_R(u)\right],
$$

where $z_f$ is the active-floor support height. Tabletop uses nearest-neighbor
resampling for this scene map. The footprint support height and support score
are

$$
h_t^{\mathrm{sup}}(a,R)=\widehat z_t(a,R)+b_{\min},
\qquad
Q_t(a,R)=\frac{1}{|\mathcal K_R^t(a)|}
\sum_{u\in\mathcal K_R^t(a)}
\mathbf 1[\widehat z_t(a,R)+b_R(u)-\bar h_s(a+u)\le\epsilon_f],
$$

with $\epsilon_f=0.02$ m. Over the candidates with valid floor metrics,

$$
H_t(a,R)=\frac{h_t^{\mathrm{sup}}(a,R)-h_{t,\min}^{\mathrm{sup}}}
{\max(h_{t,\max}^{\mathrm{sup}}-h_{t,\min}^{\mathrm{sup}},10^{-6}\ \mathrm m)}.
$$

Only kernel cells near the object's bottom and inside the nominal target
rectangle contribute to the floor-filling footprint:

$$
\mathcal A_t(a,R)=\left\{a+u:\ u\in\mathcal K_R^t(a),
\ b_R(u)\le b_{\min}+\epsilon_f,
\ t_R(u)\ge b_{\min},
\ a+u\in\mathcal T_{xy}\right\}.
$$

Let $\Omega_t$ be active-floor occupancy mapped to the proposal grid and
$\mathcal N_t$ the eight-neighbor mask computed on the occupancy grid and
then mapped to the proposal grid. With
$\mathcal U_t=\mathcal A_t\setminus\Omega_t$, the remaining area terms are

$$
O_t=\frac{|\mathcal U_t|}{|\mathcal A_t|},
\qquad
F_t=\frac{|\mathcal U_t|}
{\max(1,\max_{(a',R')\in\mathcal C_t}|\mathcal U_t(a',R')|)},
\qquad
G_t=\frac{|\mathcal U_t\cap\mathcal N_t|}{|\mathcal A_t|}.
$$

Here $\mathcal C_t$ is the set with accepted floor metrics before ranking.
Empty footprints or zero unfilled area are rejected. If occupancy exists,
the separate connectivity check requires at least one cell in
$\mathcal A_t\cap\mathcal N_t$. Tabletop's $G_t$ counts only unfilled
neighboring cells, while excavator's $G$ counts its complete coarse footprint.

For tabletop boundary scoring, let

$$
l_t=o_{xy}-\tfrac12(W,\ell)+0.3e_{xy},
\qquad
u_t=o_{xy}+\tfrac12(W,\ell)-0.3e_{xy},
\qquad
c_t=\frac{l_t+u_t}{2},\quad r_t=\frac{u_t-l_t}{2}.
$$

At the enumerated valid anchors, the boundary term is

$$
B_t(a)=\left\|(a-c_t)\oslash\max(r_t,10^{-12}\ \mathrm m)\right\|_\infty,
$$

with the maximum applied componentwise. This proposal heuristic uses the
nominal base rectangle and does not apply the excavator's contact-height taper
correction. Using the definitions above explicitly, the tabletop score is

$$
S_t(a,R)=-3H_t(a,R)+2Q_t(a,R)+O_t(a,R)
+3F_t(a,R)+G_t(a,R)+1.5B_t(a).
$$

For the **continuous integral counterpart**, let $\mathcal P_R^t(a)$ be
the projected kernel clipped to the observation domain. Replace the maximum
in $\widehat z_t$ by a supremum over this domain, retain
$h_t^{\mathrm{sup}}=\widehat z_t+b_{\min}$, and write

$$
Q_t(a,R)=
\frac{\displaystyle\int_{\mathcal P_R^t(a)}
\mathbf 1[\widehat z_t+b_R(u)-\bar h_s(a+u)\le\epsilon_f]\,d^2u}
{\displaystyle\int_{\mathcal P_R^t(a)}1\,d^2u}.
$$

Interpret $\mathcal A_t(a,R)$ as the continuous region satisfying the same
bottom-height, upper-height, and target-domain conditions above. With
$\chi_t$ the resampled occupancy indicator and $\nu_t$ the resampled
eight-neighbor indicator, define

$$
\begin{aligned}
A_t(a,R)&=\int_{\mathcal A_t(a,R)}1\,d^2p,\\
U_t(a,R)&=\int_{\mathcal A_t(a,R)}[1-\chi_t(p)]\,d^2p,\\
L_t(a,R)&=\int_{\mathcal A_t(a,R)}[1-\chi_t(p)]\nu_t(p)\,d^2p.
\end{aligned}
$$

Consequently,

$$
O_t=\frac{U_t}{A_t},
\qquad
F_t=\frac{U_t}
{\max\left(\Delta A_t,
\max_{(a',R')\in\mathcal C_t}U_t(a',R')\right)},
\qquad
G_t=\frac{L_t}{A_t},
$$

where $\Delta A_t$ is the proposal grid's uniform area weight, representing
one sample in the denominator safeguard. These expressions preserve
tabletop's restriction of the frontier numerator to **unfilled** area.
$H_t$, $B_t$, and their score weights retain their definitions above.

It deterministically sorts candidates by descending score and interleaves
queues associated with different gap subregions and orientations. It then
applies a **3 mm** minimum XY separation within each orientation. With
lower-floor priority disabled, it instead sorts by increasing scene height.

The common generator interleaves proposals across objects and supports
resuming from an attempt offset. This differs from excavator's fresh random
draws. With lower-floor priority enabled, tabletop's `planar_score` records
the complete six-term floor score and determines proposal order. The root
ranking bonus instead reads the separate `height_map_score_normalized`
diagnostic, computed from footprint support height. With floor ranking
disabled, `planar_score` is the negative raw anchor height.

The scan-relative solved-orientation filter uses a 45° bound on the XY
components of the relative rotation vector and rejects upside-down changes.
It admits a documented exception for solved poses that fill the active lower
layer. These are rotation-vector tests, not separate Euler roll/pitch limits.

### 7.4 Removal of unused sampling code

After alignment, neither platform used the deterministic height-map sampler,
its `FloorProposalRanker`, or the tabletop legacy `ActionBuilder` adapter.
Those implementations and their obsolete exports, factories, and six-term
weights have been removed. Tests specific to those removed implementations
were removed; measured-orientation coverage now exercises the scored sampler.

Two uncalled private helpers, `active_layer_fill_metrics` and
`target_structure_xy_overlap`, and an unused smoothing import were also
removed. The active `ActiveFloorPolicy` and `TargetXYOverlapFilter` continue
to implement those checks. The unconsumed `lower_floor_fill_reject_stacked`
YAML key and its legacy constants were removed from both profiles.

Before/after verification compared **480 arrays** covering both platforms,
both orientation modes, empty and placed scenes, normal and relaxed scoring,
seeded draws, mutation-score lookup, and generation retries. They matched
bit-for-bit. Removing the ignored YAML key changes serialized settings, so
the existing strict saved-context comparison can reject an earlier run even
though the effective proposal behavior is unchanged.

The current five-term score, floor geometry, standalone scored-sampler floor
context, and conditional root fallback remain active. In particular, the
`planar.floor.footprint_scale` controls rectangular footprints; connectivity
requires at least one neighboring occupied cell. The U-shape pocket
filter and its exclusive helpers have been removed. Historical score maps
remain viewable when that filter was disabled; the viewer explicitly rejects
runs that enabled it, since their candidate masks cannot be reproduced.

## 8. Initial vertical translation

After selecting XY and orientation, the initializer temporarily sets
translation Z to zero, transforms the chosen object's geometry, and computes
its bottom height $b_i$ and XY bounding box. For existing placed objects,
it computes transformed bounding boxes at their current poses.

Let $\mathcal S(a,R)$ be placed objects whose padded XY bounds overlap the
candidate. The reference support height is

$$
h_{\mathrm{ref}}=\max\left(z_{\mathrm{base}},
\max_{j\in\mathcal S(a,R)}z_j^{\max}\right),
$$

and the initial translation is

$$
z_i^0=h_{\mathrm{ref}}-b_i+c.
$$

The clearance $c$ is 0.15 m for excavator and 0.015 m for tabletop. This
places the candidate above the highest potential support detected by the
bounds heuristic. It can overestimate support under a concavity or a broad
bounding box; it does not establish an actual contact location.

Excavator uses PLY vertices when available. Its 0.12 m padding expands both
footprints, so the overlap inequalities have an effective 0.24 m allowance
per axis. Tabletop uses component vertices and its overlap inequalities use a
single 0.012 m allowance. The two padding parameters therefore have different
operational meanings despite having similar names.

The bottom-kernel height estimate used in scoring is not substituted directly
into this initialization formula. Every fresh proposal and root mutation goes
through the initializer before pose optimization.

## 9. Native refinement and action-build acceptance

The initial pose is supplied to native DSF pose generation. Both current
platform profiles use four continuation stages at
$t\in\{0,1/3,2/3,1\}$, with platform-specific scheduled objective weights.
The refined pose can change translation and rotation beyond the discrete
initial orientation bank.

Action construction applies the following current checks:

| Check | Excavator | Tabletop |
|---|---|---|
| Invalid native result | Rejected | Rejected |
| Solver convergence flag | Not required by the generic filter | Not required by the generic filter |
| Gap residual | $c_{\mathrm{gap}}\le0.005$ | $c_{\mathrm{gap}}\le0.003$ |
| Normalized equilibrium residual | $c_{\mathrm{feq}}<1$ | $c_{\mathrm{feq}}<1$ |
| Target gate | Solved object's XY AABB overlaps nominal target rectangle | Estimated volumetric containment at least 0.5 |
| Strict floor filter | Enabled, with the conditional search fallback | Disabled |
| Scan-relative orientation filter | Disabled in current PCA mode | Enabled |

The native adapter normalizes the maximum L1 net-wrench norm by the maximum
object mass in the inventory. The gap value is the native residual, not a
claim that every geometric gap equals this threshold. Convergence status and
physical validity are separate checks; CUDA can still reject unusable native
results even when the generic convergence requirement is disabled.

The target gate is applied after native refinement, as are these floor and
orientation filters. Dynamic settling, noisy-placement trials, and final
motion planning provide additional evaluation beyond this initial-pose module.

## 10. Root population refinement

The implementation class is named `CEMRootActionGenerator`. Its operation is
best described in a methods section as **elite-guided population refinement
with fixed Gaussian XY/yaw mutations and fresh proposals**. It does not fit an
adaptive Gaussian mean or covariance to the elites.

It first requests a fresh population, removes duplicates, and ranks accepted
native-refined candidates. On each subsequent round, it limits parent
contributions per object, selects an elite list, and chooses mutation parents
uniformly from that list. It perturbs the parent's saved **initial** pose:

$$
a'=a+\eta_{xy},\quad
\eta_{xy}\sim\mathcal N(0,\sigma_{xy}^2I_2),
\qquad
R'=R_z(\eta_\psi)R,\quad
\eta_\psi\sim\mathcal N(0,\sigma_\psi^2).
$$

It recomputes initial Z and reruns native pose generation and filters. The
same-object identity is retained. Mutation introduces continuous XY and yaw
variation around the structured seeds; it does not independently sample Z,
roll, or pitch. Gaussian mutations are not clamped to the original anchor
mask before solving; solved-pose filters provide the subsequent gate.

Newly built mutations and fresh proposals are combined with retained
candidates, deduplicated, and ranked. If the fresh quota cannot be filled and
parents exist, additional parent mutations can replace missing fresh attempts.
Population sizes are upper limits because proposal availability, rejection,
deadlines, and per-object caps can reduce the accepted pool.

For the standard composition, the root ranking score is

$$
J_{\mathrm{root}}=
-w_s\min\left(\frac{\max(c_{\mathrm{feq}},0)}{\tau_e},1\right)
+w_c C_i+w_h\widetilde S_i,
$$

where $C_i$ is estimated object containment, $\tau_e=1$, and
$\widetilde S_i$ is the normalized proposal score bonus. The weights
$(w_s,w_c,w_h)$ are **(2, 2, 1)** for excavator and **(0.5, 2, 1)** for
tabletop. This ranking occurs before simulation and differs from the complete
transition reward. Both platforms now supply the normalized scored-map value.

For a fresh candidate on either platform, the normalized bonus is

$$
\widetilde S_i=
\operatorname{clip}\left(
\frac{S_i-S_{\min}}{\max(S_{\max}-S_{\min},10^{-12})},0,1\right),
$$

where the extrema are over that object's retained anchor/orientation pairs.
Before alignment, tabletop instead normalized footprint support height
against extrema of its nearest-neighbor resampled scene map. A flat map then
gave zero bonus; the current shared scorer can distinguish boundary and
reference-distance features even when the scene height is constant.

The object containment term is estimated on the target's voxel-center set
$\mathcal P_T$. Let $r_v$ be voxel spacing, $(R_i,t_i)$ the **refined**
pose, and $\mathcal V_i$ the union of local convex-component volumes. Then

$$
\widehat V_{i\cap T}=r_v^3
\sum_{p\in\mathcal P_T}
\mathbf 1[R_i^{\mathsf T}(p-t_i)\in\mathcal V_i],
\qquad
C_i=\operatorname{clip}\left(
\frac{\widehat V_{i\cap T}}{V_i},0,1\right).
$$

The code uses $V_i$ equal to the sum of component volumes and voxel spacing
$r_v=0.03$ m for excavator or $0.003$ m for tabletop. Points belonging to
multiple components contribute once to the intersection count, while
overlapping component volumes can contribute more than once to $V_i$.

The corresponding **volume integral** is

$$
V_{i\cap T}=\int_{\mathcal T}
\mathbf 1[R_i^{\mathsf T}(p-t_i)\in\mathcal V_i]\,d^3p,
\qquad
C_i^{\mathrm{int}}=\operatorname{clip}\left(
\frac{V_{i\cap T}}{V_i},0,1\right).
$$

Here $\mathcal T$ is the three-dimensional target volume and
$d^3p=dp_x\,dp_y\,dp_z$. The voxel sum above is its uniform-volume
quadrature with weight $r_v^3$. To match the implementation, keep the
component-volume-sum denominator $V_i$, including its treatment of component
overlap.

For a mutation on either platform that does not match an already sampled pose, the
proposal bonus falls back to the nearest XY cell of the object's cached score
map. That map stores the best score among the discrete orientations at each
cell; it does not recompute a bottom kernel for the newly perturbed yaw.
Before alignment, tabletop mutations queried raw height at a nearest cell
of the original **128 × 128** map, while fresh proposals used footprint
support height on the resampled **32 × 32** map. That differing feature and
normalization path is no longer used by the active tabletop composition.
The standard root ranking reads explicit `height_map_score_normalized`
diagnostics, which now carry normalized scored-map values on both platforms.
Native refinement still evaluates the actual mutated pose. These bonus
lookups should be described as approximate guidance.

| Root parameter | Excavator `mcts_cuda.yml` | Tabletop `mcts_cpu.yml` and `mcts_cuda.yml` |
|---|---:|---:|
| Population | 96 | 96 |
| Elite-list limit | 32 | 24 |
| Total rounds, including initial population | 3 | 3 |
| Final retained-candidate limit | 32 | 12 |
| XY mutation standard deviation | 0.20 m | 0.01 m |
| Yaw mutation standard deviation | 5° | 5° |
| Fresh fraction in later rounds | 0.50 | 0.30 |
| XY duplicate threshold | 0.03 m | 0.003 m |
| Angular duplicate threshold | 5° | 5° |
| Per-object cap for elites and final retention | 2 | 8 |
| Proposal bonus weight | 1 | 1 |
| Seed | 0 | 0 |

Duplicate detection compares object identity, initial XY distance, and the
**full quaternion angular distance**. Although the configuration calls the
last value `duplicate_yaw_threshold_degrees`, the implementation does not
compare yaw alone. It also does not compare initial Z.

Three configured rounds mean one initial population plus two mutation/fresh
rounds. In normal parent-supported rounds, the fresh quota is 48 for excavator
and 29 for tabletop, with the remainder allocated to mutation. Root retries
increment the mutation seed and exclude earlier root proposals.

## 11. CPU/CUDA placement and scene-identification initialization

Grid construction, orientation generation, score evaluation, and root mutation
are Python-side operations. Excavator's selected CUDA profile batches the
native pose-generation and simulation work. It does not define a different
initial orientation bank or XY proposal distribution. Numerical acceptance,
runtime, and deadline effects can still alter the resulting search.

The user's open native files refer to separate operations:

- `src/sceneid/evaluation.cpp` evaluates scan agreement and screens DSF gaps
  for supplied poses.
- `src/cuda/queries.cu` evaluates support, contact, and related geometric
  queries used by native computation.

`SceneIdentifier::solve()` initializes from the current poses in its canonical
scene and sets local pose increments to zero. If the starting scene penetrates,
it first attempts physical restoration and uses the restored poses as the
chart seed. It then alternates scan fitting and physical refinement. This is
local scene estimation from a supplied scene, rather than a global bank of
placement hypotheses.

Upstream perception has its own initial registration hypotheses. Excavator
field registration constructs 64 Euler-angle combinations from
$\{0,90,180,270\}^3$, some of which represent equivalent rotations;
in-hand registration constructs four X-axis rotations relative to the supplied
pose. Tabletop's field matcher uses a 24-orientation bank, centers each
hypothesis against the observation, and brings its first-percentile height to
the table plane. Providing an explicit initial transform bypasses that bank.
These hypotheses estimate observed object poses and should be described in a
perception subsection rather than counted as placement seeds.

## 12. Paper-ready methods draft

The following passage describes the current excavator implementation. Replace
the parameter values if the experiments used different effective profiles.

> We generate initial placement hypotheses by combining a finite set of
> geometry-derived orientations with scene-aware planar anchors. For each
> available stone, principal-component analysis aligns its longest and shortest
> axes with the horizontal X and vertical Z axes, respectively. Two opposite
> roll variants and four cardinal yaw angles provide up to eight distinct
> initial orientations. We evaluate these orientations over a regular planar
> grid restricted by a target-region mask. Orientation-dependent lower-surface
> kernels estimate contact coverage against the current stack height map.
> Candidate scores combine contact height, approximate support coverage,
> unfilled floor area, target-boundary proximity, and distance from a reference
> point. A separate footprint-connectivity mask excludes disconnected seeds. Highest-scoring elites and Boltzmann draws select promising hypotheses while
> retaining stochastic exploration.

> We initialize vertical translation above the highest placed-object bounding
> box whose padded planar footprint overlaps the candidate. Each hypothesis is
> then refined by a native DSF placement optimizer and screened using
> equilibrium, gap, target-overlap, and active-floor checks. At the root of the
> placement search, an elite-guided population refinement stage adds fixed-scale
> Gaussian perturbations to initial XY coordinates and yaw while retaining
> fresh grid proposals. Heights and refined placements are recomputed for every
> mutation. Per-object caps and pose deduplication preserve diversity across
> stones. Accepted placements are subsequently evaluated by simulation within
> MCTS. The excavator policy prioritizes filling the active lower layer and
> permits a conditional relaxation of floor-fill restrictions when ordinary
> root selection fails.

## 13. Reproducibility and interpretation

Record the actual experiment configuration, input scene, asset hashes,
random-number seeds, backend versions, and effective time budget. A fixed
seed makes the random draws reproducible for the same call sequence; changes
in retries, object order, rejection, or deadlines can change that sequence.

The current geometry and sampling impose the following limits on claims:

1. The orientation bank samples selected principal-axis faces or scan-relative
   orientations. It does not exhaustively sample SO(3), and native refinement
   does not establish global optimality.
2. Support likelihood is a geometric contact-coverage heuristic, not a
   calibrated stability probability or force-closure certificate.
3. The height map stores only the highest surface at each XY location.
   Undercuts, cavities, and disconnected vertical surfaces are simplified.
4. Different stages use global hulls, component unions, PLY bounds, and smooth
   DSFs. Their approximation errors differ and can change proposal ranking
   and refinement behavior.
5. Score normalization, floor-priority pools, and object allocation make sampling
   state dependent. The Boltzmann formula applies within each draw's pool.
6. Boundary and distance weights currently favor target edges and greater
   reference distance. Their utility should be supported by the paper's
   experiments rather than asserted from the term names.
7. Root refinement uses fixed mutation scales. Describing it as fitting a
   Gaussian distribution to elites would misrepresent the implementation.
8. Strict floor filtering includes an explicit relaxation path. The paper
   should report whether that fallback occurred in the evaluated runs.

Potential ablations include orientation-bank size, contact-kernel scoring,
elite fraction, Boltzmann temperature, floor priority, root mutations,
fresh fraction, and per-object caps. These are suggested experiments; this
document contains proposal-score sensitivity measurements below, but no new
native-placement, simulation, or end-to-end search-quality measurements.

### 13.1 Historical excavator proposal-score sensitivity

The historical results below used seven terms. Open-area and frontier scores
have since been removed. The current analysis script measures the remaining
five terms; rerunning it will not reproduce this archived report.

The diagnostic [analyze_initial_pose_scores.py](../tests/analyze_initial_pose_scores.py)
was run with the excavator configuration before its YAML layout consolidation,
all 49 configured object assets,
and the nine recorded placed-object poses in
[excavator-cuda-mcts.json](benchmarks/excavator-cuda-mcts.json). It renders
five states containing 0, 1, 3, 6, and 9 of those stones, retaining their
final recorded poses. These subsets are diagnostic scenes, rather than a
replay of historical stacking states. All configured masks, support-kernel
validity checks, connectivity gates, and lower-floor priority remain active.

The run checked **274,096** retained anchor/orientation pairs. It reconstructs
each candidate's seven weighted terms and verifies their sum bit-for-bit
against the scorer. Comparison scores are then rounded to 12 decimal places
to suppress numerical tie changes when a constant term is removed. Production
scoring does not apply that rounding. The
[full report](benchmarks/initial-pose-score-sensitivity.json) records source
and asset hashes, per-object results, and matched sampling seeds 0 through 9.
No native solve, root population refinement, or physical simulation is run.
The report retains the hashes from that run. Consolidation changes configuration
and composition hashes, but preserves the effective weights. A separate
before/after comparison verified bit-for-bit equality of 89 score-map,
seeded-selection, and shared-floor arrays for empty and placed fixture scenes.

To measure discrimination, use the standard deviation of each **weighted**
term across candidates, then average equally across available objects. A
constant contribution has zero discrimination even if its mean is large.
This statistic measures score variation; correlations between terms prevent
interpreting it as a percentage of total ranking influence.

The table uses the first nonempty sampling-priority pool for each object:
lower-floor candidates when available, otherwise all candidates. If that pool
has fewer than 32 entries, actual sampling can also draw from the higher pool.

| Placed stones | $-3H$ | $2Q$ | $O$ | $2F$ | $G$ | $1.5B$ | $2D$ | Largest variation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.000 | 0.186 | 0.000 | 0.000 | 0.000 | 0.352 | 0.204 | Boundary |
| 1 | 0.112 | 0.248 | 0.004 | 0.223 | 0.089 | 0.272 | 0.076 | Boundary |
| 3 | 0.062 | 0.210 | 0.009 | 0.221 | 0.055 | 0.345 | 0.066 | Boundary |
| 6 | 0.171 | 0.327 | 0.009 | 0.214 | 0.093 | 0.109 | 0.044 | Support |
| 9 | 0.865 | 0.218 | 0.207 | 0.437 | 0.101 | 0.452 | 0.206 | Height |

Across the complete candidate banks, height has the largest average weighted
variation in all four occupied states. Lower-floor priority removes much of
that variation before the first weighted draw. In the nine-stone state, none
of the 40 remaining objects has a candidate whose estimated contact height
is within 0.20 m of the active floor, so sampling uses the higher pool. Active
floor occupancy is still only 0.566. This is a proposal-geometry limitation
that changing score weights alone cannot resolve.

Only $H$ and $Q$ directly evaluate the scene height map. $O$, $F$, and $G$
use active-floor occupancy, while $B$ and $D$ add geometric position
preferences. The seven terms therefore represent three different sources of
guidance.

The removal comparisons keep the candidate bank fixed. Rank correlation is
Spearman correlation within the first nonempty priority pool. Top-32 overlap
uses only objects with more than 32 candidates in that pool, so retaining
every member of a small pool does not inflate the result. The report also
compares populations drawn by the sampler's regional and Boltzmann selection;
matched-seed population overlap describes sampling sensitivity rather than
physical quality or an unchanged probability distribution.

The preliminary simplification order is:

1. **Remove or merge open area $O$ first.** In occupied-floor scoring, $O$
   and $F$ share the numerator $n_U$, with different denominators. When
   footprint sample count $n_A$ is
   constant, they are proportional. Their mean per-object correlation over
   the complete occupied candidate banks is 0.89–0.96. In the 1-, 3-, and
   6-stone priority pools, $O$ is nearly constant; removing it leaves ranking
   correlation above 0.998. At nine stones it still changes some preferred
   candidates, so merging also requires checking the remaining fill weight.
2. **Ablate reference distance $D$ next.** The current reference is the target
   origin, so this is an additional edge preference without an independently
   specified reach or robot-distance objective. It overlaps with $B$, while
   using a different distance geometry. Its influence is substantial on an
   empty floor, where $H=0$, $O=1$, and $F=G=0$. Removing it will change the
   preferred first-placement locations.
3. **Ablate the frontier bonus $G$ after that.** The connectivity gate already
   requires at least one neighboring footprint cell. Removing the score
   bonus preserves that gate but removes the preference for broader
   adjacency. This is a distinct preference, so it requires a physical
   quality comparison before removal.
4. **Retain $H$, $Q$, and one fill measure initially.** They distinguish
   supporting height, approximate contact coverage, and target-floor filling.
   Retain $B$ during the first simplification because it is the strongest
   spatial preference in the empty and early priority pools. Its removal is
   a separate experiment on the placement strategy.

A concrete four-term experimental score is

$$
S_4=-3H+2Q+2F+1.5B.
$$

The more aggressive three-term comparison is $S_3=-3H+2Q+2F$. The mean
top-32 overlap with the current seven-term score is:

| Placed stones | Objects with more than 32 priority candidates | $S_4$ overlap | $S_3$ overlap |
|---|---:|---:|---:|
| 0 | 49 | 33.7% | 12.4% |
| 1 | 29 | 94.6% | 75.6% |
| 3 | 28 | 93.4% | 73.0% |
| 6 | 19 | 98.5% | 97.7% |
| 9 | 40 | 77.5% | 56.3% |

Removing $D$ alone retains 33.7% of the empty-floor top 32, illustrating its
effect on first-placement positions. On an empty floor, $S_3$ has no XY
score preference, so regional draws determine spatial coverage. Gap diversity
also applies once occupied-floor gaps exist. These are diagnostic candidates,
not changed repository defaults. Full placement comparisons should record accepted
native solves, root floor-fill fallback frequency, target coverage, failed
settling/noisy-placement trials, and search outcome under the same backend,
input states, seeds, and time budgets. Removing a proposal term can also
change the normalized root bonus described in Section 10.

Reproduce the sensitivity report with the project Python and model assets:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tests/analyze_initial_pose_scores.py \
  --output docs/benchmarks/initial-pose-score-sensitivity.json
```

## 14. Implementation references

Links are relative to this document so they remain usable from the repository.
The native references assume the sibling `stacking-core` checkout.

| Component | Source and relevant symbol |
|---|---|
| Excavator composition | [composition.py](../apps/excavator/src/stacking_excavator/composition.py): `build_orientation_provider`, `build_action_generator`, `build_floor_policy` |
| Tabletop composition | [composition.py](../apps/tabletop/src/stacking_tabletop/composition.py): `build_orientation_provider`, `build_floor_policy` |
| Shared search composition | [search.py](../packages/stacking-common/src/stacking_common/workflows/search.py): `build_mcts`, `_build_state_evaluators`, `build_action_generator` |
| Orientation bank | [orientation.py](../packages/stacking-common/src/stacking_common/actions/orientation.py): `canonical_principal_axis_rotation`, `PrincipalAxisOrientationProvider`, `ScanAlignedOrientationProvider` |
| Excavator sampler and allocation | [scored.py](../packages/stacking-common/src/stacking_common/actions/scored.py): `ScoredPlanarPoseSampler`, `ScoredPlacementActionGenerator` |
| Excavator scoring and draws | [_scored_floor.py](../packages/stacking-common/src/stacking_common/actions/_scored_floor.py): `score_xy_debug_map`, `target_anchor_bounds`, `sample_scored_candidates` |
| Bottom kernels | [_support_maps.py](../packages/stacking-common/src/stacking_common/actions/_support_maps.py): `bottom_height_kernel`, `support_likelihood_maps_by_kernel` |
| Shared current sampler | [scored.py](../packages/stacking-common/src/stacking_common/actions/scored.py): `ScoredPlanarPoseSampler`, `ScoredPlacementActionGenerator`; [search.py](../packages/stacking-common/src/stacking_common/workflows/search.py): `build_action_generator` |
| Historical tabletop sampler | Removed; equations and selection procedure are documented in Section 7.3 |
| Floor geometry | [floor.py](../packages/stacking-common/src/stacking_common/actions/floor.py): `ActiveFloorPolicy` |
| Tabletop floor values | [safety.py](../apps/tabletop/src/stacking_tabletop/planning/safety.py): `TabletopPlacementSafety` |
| Excavator height | [ply_height.py](../packages/stacking-common/src/stacking_common/actions/ply_height.py): `PlyHeightInitializer` |
| Tabletop height | [initialization.py](../packages/stacking-common/src/stacking_common/actions/initialization.py): `MeshHeightInitializer` |
| Build pipeline | [generator.py](../packages/stacking-common/src/stacking_common/actions/generator.py): `PlacementActionGenerator` |
| Pose refinement | [core_solver.py](../packages/stacking-common/src/stacking_common/actions/core_solver.py): `CorePlacementPoseSolver` |
| CUDA refinement | [cuda_solver.py](../packages/stacking-common/src/stacking_common/actions/cuda_solver.py): `CudaPlacementPoseSolver` |
| Acceptance filters | [filtering.py](../packages/stacking-common/src/stacking_common/actions/filtering.py): `PosegenFeasibilityFilter`, `ScanOrientationFilter`, `TargetContainmentFilter` |
| Root population | [cem.py](../packages/stacking-common/src/stacking_common/actions/cem.py): `CEMRootActionGenerator`, `_duplicate` |
| Root prior score | [policies.py](../packages/stacking-common/src/stacking_common/search/policies.py): `PlacementPriorScorer` |
| Containment estimate | [overlap.py](../packages/stacking-common/src/stacking_common/environment/overlap.py): `VoxelOverlapEvaluator`; [object.py](../packages/stacking-common/src/stacking_common/domain/object.py): `StackObject.volume`, `contains_local_points` |
| Floor-fill fallback | [search.py](../packages/stacking-common/src/stacking_common/search/mcts/search.py): `_run_floor_fill_fallback` |
| Height-map domain | [heightmap.py](../packages/stacking-common/src/stacking_common/visualization/heightmap.py): `HeightMapRenderer` |
| Excavator parameters | [action_generation.yml](../apps/excavator/config/action_generation.yml), [environment.yml](../apps/excavator/config/environment.yml), [inventory.yml](../apps/excavator/config/inventory.yml), [mcts_cuda.yml](../apps/excavator/config/mcts_cuda.yml), [posegen.yml](../apps/excavator/config/solvers/posegen.yml) |
| Tabletop parameters | [action_generation.yml](../apps/tabletop/config/action_generation.yml), [environment.yml](../apps/tabletop/config/environment.yml), [inventory.yml](../apps/tabletop/config/inventory.yml), [mcts_cpu.yml](../apps/tabletop/config/mcts_cpu.yml), [mcts_cuda.yml](../apps/tabletop/config/mcts_cuda.yml), [posegen.yml](../apps/tabletop/config/solvers/posegen.yml) |
| Excavator registration seeds | [pipeline.py](../apps/excavator/src/stacking_excavator/perception/pipeline.py): `orientation_seeds` |
| Tabletop registration seeds | [assignment.py](../apps/tabletop/src/stacking_tabletop/perception/assignment.py): `ORIENTATION_BANK`, `initial_table_transform` |
| Native scene-ID initialization | [consensus.cpp](../../stacking-core/src/sceneid/consensus.cpp): `SceneIdentifier::solve`; [scan.cpp](../../stacking-core/src/sceneid/scan.cpp): `validate_and_resolve_poses` |
| Native scene-ID evaluation | [evaluation.cpp](../../stacking-core/src/sceneid/evaluation.cpp): `evaluate_sceneid`, `screen_sceneid_gaps` |
| CUDA geometry queries | [queries.cu](../../stacking-core/src/cuda/queries.cu) |

Relevant existing test coverage includes
[test_excavator_action_parity.py](../tests/test_excavator_action_parity.py),
[test_floor_score_maps.py](../tests/test_floor_score_maps.py),
[test_action_generation.py](../tests/test_action_generation.py),
[test_excavator_floor_first.py](../tests/test_excavator_floor_first.py),
[test_tabletop_scored_sampling.py](../tests/test_tabletop_scored_sampling.py),
[test_tabletop_scan_orientation.py](../tests/test_tabletop_scan_orientation.py),
and [test_cuda_placement.py](../tests/test_cuda_placement.py). These files cover
seeded sampling parity, orientation construction, support footprints, height
initialization, root refinement, floor fallback, and batched refinement. They
are source references; the score diagnostic in Section 13.1 does not execute
these tests or run native-placement experiments.

## 15. Source snapshot

The HEADs and working-tree description below record the **original inspection
snapshot**. They are historical provenance, not the live checkout state.

- `stacking-apps` base HEAD: `3c2d3328e5b8552a21666434d67ea54b3813cd68`.
- `stacking-core` base HEAD: `aee2d5bb0269d8f160ec442230725cf5d7e415df`.

Both checkouts contained working-tree changes at that inspection. The excavator
action, inventory, and CUDA MCTS profiles and the common MCTS search file
had local changes; the native scene-identification sources were untracked
in the inspected core checkout. The HEAD identifiers alone therefore do
not specify the implementation summarized here. The following SHA-256
manifest identifies the referenced implementation and configuration files
as read for this document. Store the source and effective experiment
profiles with the paper artifacts to preserve reproducibility.

The score equations were expanded and checked on 2026-10-06. The subsequent
configuration consolidation changed the YAML layout and floor-policy interface
without changing the score equations or effective weights. The manifest
retains the original implementation and native source hashes; the native
scene-identification subsection describes that original snapshot.
The 2026-10-07 tabletop alignment subsequently changes its effective scoring
and selection strategy as described in Section 7.2; its earlier equations
are retained in Section 7.3. These changes are not represented by the original
manifest below.

```text
613f6f5753aab0fa42fb0ff55541c178438e16803141d918b32c4e2879072540  stacking-apps/apps/excavator/config/action_generation.yml
78fda3822f4f2fea563d8d084b83a4be59b0c54daa358af668e93aa13550d14d  stacking-apps/apps/excavator/config/environment.yml
c217ef2b227d131a5c5e0194b8548a057dced07892251141e69a9da616f64092  stacking-apps/apps/excavator/config/inventory.yml
6973a22ac546247f186b1f27a9e1161615e13b3c6ecb0bc5c075286ac27f4da6  stacking-apps/apps/excavator/config/mcts_cuda.yml
f419b1c2dfd778b9b129db0700c54636cd1d0175660048d3e6be947240db1624  stacking-apps/apps/excavator/config/solvers/posegen.yml
485735e23e4f69e77f9cbdf342681f4ca17d6d524868f62b7c4062443e3e2b42  stacking-apps/apps/excavator/src/stacking_excavator/composition.py
0a83664633a1fbcc0e52f76020e49c6bd7ed32988ed9c0da1c771611655c6b04  stacking-apps/apps/excavator/src/stacking_excavator/perception/pipeline.py
3d700daa5091f8da190cc660226659178059f589c391919617ba0e400cad96c0  stacking-apps/apps/tabletop/config/action_generation.yml
3f0e1cc58215e8e7699431551a567e7d607d0d629329173c4c8d0b880065a95f  stacking-apps/apps/tabletop/config/environment.yml
e2ccec0219b790e9422835ed650e0122dac2ada7c39747c9cf293bd0f4a0015e  stacking-apps/apps/tabletop/config/inventory.yml
10fface4e89c86dd23dbe725121b56cccfd417580927647f90eeae9de1ed15d5  stacking-apps/apps/tabletop/config/mcts.yml
3bb31fe1c327981e965d2dd135230b9cbef5575f7730305eba79272d51e4b86f  stacking-apps/apps/tabletop/config/solvers/posegen.yml
832aed442535ef56b79aa31e5faeda855141ceec4fc1cad5f6da151e6edb9b7d  stacking-apps/apps/tabletop/src/stacking_tabletop/composition.py
aaebbe2d47786fb9c86b43953a048cf58247fb6db76206aa4807c852c1e71a52  stacking-apps/apps/tabletop/src/stacking_tabletop/perception/assignment.py
1c9efdb2bcd74ca5e96543d64bf17538016b0ee8cbba145fdcda88a8764aa36f  stacking-apps/apps/tabletop/src/stacking_tabletop/planning/safety.py
29036fa58d9ae806044822bff579ef2ced2399f56558f43e193db5346983fe03  stacking-apps/packages/stacking-common/src/stacking_common/actions/_scored_floor.py
7261f73b1aaa563c7cfe85c8cac985a560ce1658fc0baf92d80e3eb785d2e566  stacking-apps/packages/stacking-common/src/stacking_common/actions/_support_maps.py
01c4d933714fa1f947b9351926af9f6357a136324218b3e3d18916f85e3adbf6  stacking-apps/packages/stacking-common/src/stacking_common/actions/cem.py
06f9f0eefec117ea9c1aa7120a524c240533d1dce3ba6184678462c3664d9de8  stacking-apps/packages/stacking-common/src/stacking_common/actions/core_solver.py
2b9274f9eff920db8dd9b1e9226fbf6e2f4fc40c4a369b931589ad65622d6b0a  stacking-apps/packages/stacking-common/src/stacking_common/actions/cuda_solver.py
483a995679d35d7f67bb8858617c006b4fa5bbc968cb9de46c0e84a407292970  stacking-apps/packages/stacking-common/src/stacking_common/actions/filtering.py
1b46dd34e214f462fd3236458ca927c47608c1d9f880489ca076c6587c07b7c4  stacking-apps/packages/stacking-common/src/stacking_common/actions/floor.py
c2f7f9727921163669e5cd09ce06b84fd1d6e76a5ea37257277280a3cd594a46  stacking-apps/packages/stacking-common/src/stacking_common/actions/generator.py
ff2b9c2f2838464ab9307263ba008df3631e4551d9e0d264e099b1f3da4ba376  stacking-apps/packages/stacking-common/src/stacking_common/actions/initialization.py
51a4d4012b4b96da31ef69787c0af11253e70f93659602c1f7c39e4e5d88c3a6  stacking-apps/packages/stacking-common/src/stacking_common/actions/orientation.py
ccad336807408104cabe11ae622092426df9b94f8312a670d2039d8be5f6cd61  stacking-apps/packages/stacking-common/src/stacking_common/actions/planar.py
c541358b3b63a591af4d8c5b7563178079fd0f68e6ce2ecbbb5fa6103be5af58  stacking-apps/packages/stacking-common/src/stacking_common/actions/ply_height.py
376a5235a32fd8aba012a955dbacdf7f01c296048e635ba2cf015a94f9294acd  stacking-apps/packages/stacking-common/src/stacking_common/actions/scored.py
638602f2a0a0aaeb1a6d3357c1dda0a7327308a193667edba4d1c9b670fd173c  stacking-apps/packages/stacking-common/src/stacking_common/domain/object.py
b27ea439a6b2249b5641028a4a4e3b6ad3f3bd3cfc3ed575c5f77144cbe01679  stacking-apps/packages/stacking-common/src/stacking_common/environment/overlap.py
ca4482b8b39828dc291792a15b621197ab8ac0f53a594076439a46c047219d57  stacking-apps/packages/stacking-common/src/stacking_common/search/mcts/search.py
a8b0947351191f53b0450728e4089d2447f6955c430fa92dc6411f82fa73c5ba  stacking-apps/packages/stacking-common/src/stacking_common/search/policies.py
85df6ff33847120d618d3763db6993ee1bb69d2febd3311af08f1837088f27f5  stacking-apps/packages/stacking-common/src/stacking_common/visualization/heightmap.py
ae8c0367b51ab31735f74900a8ee77e5c9ff3cb786bbf2e9db3de0c9216a44a9  stacking-apps/packages/stacking-common/src/stacking_common/workflows/search.py
e75dcfb08e1614b87324e918bc59146d8f3c64acb4f083a8f8e6757fa9f35abe  stacking-core/src/cuda/queries.cu
6b3c0b6a1478ea5741ea12d07c7da7b616f5621b3c4f737d5dde89882f2b181f  stacking-core/src/sceneid/consensus.cpp
d5ec9e1b8d1afa789bf112e90938d6967b5805400bf9e08b658d96eeed0b178d  stacking-core/src/sceneid/evaluation.cpp
4e6ba296db1f590d584eab21d7c6643c5540162c25cd777884c32680e71ee5b5  stacking-core/src/sceneid/scan.cpp
```
