# Initial placement sampling for automated stone stacking

Methods summary for an RSS 2027 manuscript, based on the implementation
inspected on **2026-10-18**. Parameters describe repository defaults; use the
effective experiment settings in the submitted paper. The
[implementation reference](initial-pose-sampling.md) contains source links,
solver coefficients, historical comparisons, and reproducibility details.
The subsequent settling, structural, and robustness stages are described in
[MCTS simulation](mcts-simulation.md).
The complete search policy is summarized in [MCTS algorithm](mcts-algorithm.md).

## 1. Method overview

For each placement, we generate initial stone poses from the current stack
geometry and target region. A hypothesis comprises a stone identity, an XY
anchor, and an orientation from a finite bank. Its vertical translation is
computed geometrically. A native DSF placement optimizer then refines all six
pose degrees of freedom before physical evaluation in Monte Carlo tree search
(MCTS).

Fresh proposal attempts are allocated among available stones with probability

$$
P(i)=\frac{V_i}{\sum_{j\in\mathcal E}V_j},
$$

where $V_i$ is the sum of the stone's convex-component volumes and
$\mathcal E$ contains unplaced, non-banned stones below their request cap.
Each stone receives at most 32 attempts per request. This allocation gives
larger stones more proposal opportunities; the subsequent geometric score
determines where each stone is initialized.

## 2. Orientation and planar hypotheses

For excavator stones, principal-component analysis of the global convex-hull
vertices aligns the longest axis with horizontal X and the shortest with Z.
Two roll variants and four yaw angles produce up to eight distinct rotations:

$$
R=R_z(\psi)R_x(\phi)R_{\mathrm{PCA}},\qquad
\phi\in\{0,\pi\},\quad
\psi\in\{0,\pi/2,\pi,3\pi/2\}.
$$

These orientations place the shortest principal axis vertically. Tabletop preserves the
measured tilt, aligns the projected longest axis with X modulo $\pi$, and
uses two yaw variants, $0$ and $\pi$.

Both platforms render a 128 × 128 upper-surface height map from the placed
component meshes and evaluate anchors on a 32 × 32 grid. For each rotation,
let $q_R^-,q_R^+$ be the rotated hull's XY minimum and maximum offsets from
the stone origin. An anchor $a$ is target eligible when

$$
L-\gamma q_R^-\le a\le U-\gamma q_R^+,\qquad \gamma=0.9,
$$

where $L,U$ bound the nominal target rectangle. This admits limited initial
overhang and accounts for stones whose origins are offset from their centers.
Both the target mask and floor-filling footprint depend on orientation.

## 3. Geometric proposal score

The proposal score combines low contact height, approximate support coverage,
unfilled floor area, and proximity to a target boundary:

$$
S(a,R)=-3H(a,R)+2Q(a,R)+2F(a,R)+1.5B(a,R).
$$

The following integral expressions describe the spatial terms continuously.
The implementation evaluates them using uniform grid samples; $d^2u$ denotes
projected XY area, rather than area along the stone surface.

### Support coverage and contact height

Let $h(p)$ denote scene height and $b_R(u)$ the lower surface of the rotated
global convex hull over its projection $\mathcal P_R$. The estimated
translation needed to clear the sampled scene and the resulting gap are

$$
\widehat z(a,R)=\sup_{u\in\mathcal P_R}[h(a+u)-b_R(u)],\qquad
d(a,R,u)=\widehat z(a,R)+b_R(u)-h(a+u).
$$

With contact indicator $c(u)=\mathbf 1[d(a,R,u)\le\epsilon_c]$,

$$
Q(a,R)=\frac{\int_{\mathcal P_R}c(u)\,d^2u}
{\int_{\mathcal P_R}1\,d^2u},\qquad
h_c(a,R)=\frac{\int_{\mathcal P_R}h(a+u)c(u)\,d^2u}
{\int_{\mathcal P_R}c(u)\,d^2u}.
$$

$Q$ measures the fraction of the projected bottom near the scene surface.
The height term is
$H=(h_c-h_{\min})/\max(h_{\max}-h_{\min},10^{-6}\,\mathrm m)$,
with extrema taken over the stone's target/open-space/support-valid
hypotheses before connectivity masking. Thus lower contact surfaces receive
higher scores. The discrete implementation rejects empty support kernels or
incomplete scene-map coverage. Contact coverage is a geometric heuristic;
force equilibrium is evaluated during refinement.

### Unfilled area and boundary proximity

Let $\Omega$ be active-floor occupancy and
$\mathcal A(a,R)=\mathcal T_{xy}\cap[a+0.9q_R^-,a+0.9q_R^+]$ the
rectangular scoring footprint clipped to the nominal target. Its unfilled
area and normalized fill score are

$$
U(a,R)=\int_{\mathcal A(a,R)}[1-\mathbf 1_{\Omega}(p)]\,d^2p,
\qquad
F(a,R)=\frac{U(a,R)}
{\max\left(\Delta A_f,\max_{(a',R')\in\mathcal C_{\mathrm{open}}}U(a',R')\right)}.
$$

Here $\mathcal C_{\mathrm{open}}$ contains target-valid hypotheses with
positive unfilled area before connectivity/support filtering, and
$\Delta A_f$ is one floor-grid cell's area. The maximum is shared across
orientations of the same stone. An empty active floor uses $F=0$.

For boundary scoring, the target bounds are tapered inward at the estimated
contact height and inset using the rotated offsets above. Let their center
and half-span be $c_T,r_T$. Then
$B=\|(a-c_T)\oslash r_T\|_\infty$ inside valid bounds, and zero otherwise;
$\oslash$ denotes componentwise division. This rewards anchors nearer a
target edge. The configurable reference-distance term has zero weight in
both current profiles.

## 4. Floor admission and stochastic selection

Active-floor occupancy is the union of projected component footprints on a
16 × 16 cell-center grid. Starting at the target base, the policy groups
stones whose bottoms lie within a platform-specific support-height tolerance.
Once coverage reaches the completion threshold, the active height advances
to the highest top in that group.

When the active floor is occupied, ordinary proposals require positive
unfilled footprint area and at least one footprint cell in the eight-neighbor
expansion of occupancy. Connectivity is a hard admission rule. After the first
placement, proposals whose contact height lies within the active-floor
tolerance are sampled before higher proposals.

Within each pool, the sampler first selects the best
$\lceil0.25k\rceil$ hypotheses for a request of $k$ proposals, capped by pool
size. Remaining slots are drawn without replacement with

$$
P(j\mid\mathcal P)=
\frac{\exp(S_j/0.3)}{\sum_{\ell\in\mathcal P}\exp(S_\ell/0.3)},
$$

where $\mathcal P$ is the remaining pool. In the empty scene, sampling
instead starts from feasible grid neighborhoods of the four target corners,
using full-footprint center bounds. Corners are shuffled and represented
before remaining corner seeds are drawn. If no corner seeds exist, ordinary
scored sampling is used. The generator draws up to 32 proposals per allocated
stone and retains its budgeted prefix for corner or lower-floor sampling.

## 5. Vertical initialization and native refinement

The candidate is initially placed above all placed stones with overlapping,
padded XY bounding boxes:

$$
z_i^0=\max\left(z_{\mathrm{base}},
\max_{j\in\mathcal S(a,R)}z_j^{\max}\right)-b_i+c_z,
$$

where $b_i$ is the candidate's lowest vertex at zero Z translation and $c_z$
is a clearance. This conservative initialization differs from the support
height used in scoring. Excavator uses reconstructed PLY vertices when
available; tabletop uses component-mesh vertices.

The DSF optimizer refines each pose through continuation stages
$t\in\{0,1/3,2/3,1\}$, with scheduled contact, equilibrium, target, and
placement penalties. It can move XY and rotation beyond the initial bank.
Solved poses pass gap, legacy equilibrium, target, and active-floor filters;
tabletop also applies a scan-relative orientation check. Subsequent simulation
evaluates settling and placement robustness.

## 6. Root population refinement

At the MCTS root, an initial population of up to 96 refined candidates is
followed by two refinement rounds. Each later round allocates half its
attempts to fresh proposals and half to mutations of uniformly selected
elite parents. Mutations perturb the parent's **initial** XY and yaw:

$$
a'=a+\eta_{xy},\quad\eta_{xy}\sim\mathcal N(0,\sigma_{xy}^2I),\qquad
R'=R_z(\eta_\psi)R,\quad\eta_\psi\sim\mathcal N(0,(5^\circ)^2).
$$

Z is recomputed and native refinement is repeated. The implementation uses
fixed mutation scales rather than fitting an adaptive covariance. Candidates
are ranked by

$$
J=-2\operatorname{clip}(E,0,1)+2C+\widetilde S,
$$

where $C$ is voxel-estimated object containment and $\widetilde S$ is the
proposal score normalized over admitted hypotheses of that stone. For body
net force $f_b$ and moment $\tau_b$, the dimensionless equilibrium residual is

$$
E=\max_b\left[
\frac{\|f_b\|_2}{m_bg_0}
+\frac{\|\tau_b\|_2}{m_bg_0V_b^{1/3}}\right].
$$

Here $m_b,V_b$ are body mass and component-volume sum, and $g_0$ is gravity.
This ranking residual is distinct from the legacy feasibility threshold.
Mutation proposal bonuses use the nearest cached score-map cell; native
refinement evaluates the actual mutated pose. Deduplication and a two-per-stone
cap yield at most 32 retained candidates. These limits are upper bounds because
rejection and deadlines can reduce the population.

## 7. Platform parameters and scope

| Parameter | Excavator | Tabletop |
|---|---:|---:|
| Default orientation count | Up to 8 | Up to 2 |
| Support contact tolerance $\epsilon_c$ | 0.10 m | 0.02 m |
| Active-floor height tolerance | 0.20 m | 0.02 m |
| Floor completion threshold | 75% | 65% |
| Initial Z clearance $c_z$ | 0.15 m | 0.015 m |
| XY mutation deviation $\sigma_{xy}$ | 0.20 m | 0.01 m |
| XY duplicate threshold | 0.03 m | 0.003 m |
| Containment voxel spacing | 0.03 m | 0.003 m |
| Native gap acceptance threshold | 0.005 | 0.003 |

Both platforms share the proposal score, temperature, elite fraction, and
population limits. Their physical values are independently configured.
CPU and CUDA use the same Python proposal strategy; CUDA batches native
refinement without defining a different sampling distribution.

Conditional retries can replace unsuccessful corner seeding with ordinary
sampling or relax floor admission while retaining connectivity and physical
evaluation. Experiments should report whether these paths were used. The
finite orientation bank and height-map representation simplify concavities
and can leave some stones without feasible anchors; the detailed reference
records the current tabletop stone-2 case. Sampling alone establishes neither
global pose optimality nor physical stability.
