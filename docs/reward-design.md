# Reward Design

> Implementation update (2026-10-28): surface flatness now rewards the reduction
> in relief between parent and settled scenes. This dated paper draft retains
> the earlier absolute-penalty formula; see the
> [current specification](initial-pose-sampling.md#mcts-surface-flatness-reward).

논문 본문용 영문 축약 초안이다. 항별 정의, 가중치, 정규화와 실패 처리는
[Appendix 초안](reward-design-appendix.md)에 정리했다.
현재 구현 확인일은 2026-10-21이다. 본문은 현재 excavator의 활성 reward 항을
기준으로 한다. Tabletop에서 추가 사용하는 orientation 항은 appendix에 설명한다.

## 1. 본문 초안

We score each placement by balancing construction progress and geometric
quality against equilibrium error and sensitivity to disturbances. For an
action $a$ in state $s$ with simulated successor $s'$, the reward is

$$
\begin{aligned}
r(s,a,s')={}&w_p\phi_p+w_c\phi_c\\
&-w_e\psi_e-w_f\psi_f-w_r\psi_r.
\end{aligned}
$$

Here $\phi_p$ rewards the newly placed stone's volume contained within the
target, and $\phi_c$ rewards its containment fraction above an acceptance baseline.
The penalties measure residual equilibrium error ($\psi_e$), upper-surface
relief ($\psi_f$), and motion under velocity disturbances ($\psi_r$).

**Construction and geometry.** The progress term favors placements that add
more stone volume inside the target, while the containment term discourages
overhang independently of stone size. Surface regularity is measured by
normalized upper-surface relief:

$$
\psi_f=\frac{h_{\max}-\overline h}{\bar V^{1/3}}.
$$

Here $h_{\max}$ and $\overline h$ are the maximum and mean upper-surface
heights over the target footprint, including empty floor, and $\bar V$ is the
initial inventory's mean stone volume. Filling lower regions without raising
the peak reduces this penalty.

**Equilibrium and robustness.** The equilibrium penalty uses force and moment
residuals from placement optimization. Disturbance robustness is evaluated
from the nominally settled assembly by applying initial velocity perturbations
to the new stone and measuring the worst peak translation-plus-weighted-rotation
excursion across dynamic stones and trials:

$$
D=\max_{k,b,t}\left[
\|p_b^{(k)}(t)-p_b^0\|_2+
\lambda_{\mathrm{rot}}\theta(q_b^{(k)}(t),q_b^0)\right],
\qquad \psi_r=\frac{\overline D}{D_{\mathrm{ref}}}.
$$

The reference poses are those of the nominally settled assembly, and $\theta$
is the shortest rotation angle. The coefficient $\lambda_{\mathrm{rot}}$
converts rotation to equivalent displacement. The processed excursion
$\overline D$ incorporates reward clipping and nonfinite handling; rejection
uses the raw excursion and recovery outcomes. Peak measurement penalizes
transient motion even when the assembly subsequently returns near its original pose.

**Normalization and feasibility.** Volume and surface relief are normalized
using the initial inventory's mean stone volume and its associated length
scale. Equilibrium residuals use each body's weight and characteristic size;
disturbance motion uses a platform-specific reference length. Hard physical
and task constraints determine whether a transition remains selectable.
Failed transitions use a fixed failure reward during search backup. Detailed
feature definitions, normalization conventions, and weights are provided in
the appendix.

## 2. Appendix 구성안

| 항목 | 상세 내용 |
|---|---|
| Progress and containment | Contained volume, containment fraction, voxel approximation, inventory reference |
| Equilibrium | Force/moment residual, body별 정규화, legacy fallback |
| Surface geometry | Surface relief, tabletop의 추가 upward/inward orientation 항 |
| Disturbance robustness | Peak excursion, 단위·정규화, clipping과 rejection의 구분 |
| Weights and constraints | 현재 가중치, 비활성 항, 실패 backup, 플랫폼별 설정 |

전체 탐색 흐름은 [MCTS 본문 초안](mcts-algorithm.md), 외란 실험의 실행 과정은
[시뮬레이션 문서](mcts-simulation.md)를 참고한다.
