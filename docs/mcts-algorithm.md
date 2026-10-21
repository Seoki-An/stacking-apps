# Simulation-Guided Tree Search for Stone Stacking

논문 본문용 축약 초안이다. **구조물을 어떻게 탐색하고, 물리 평가를 어떻게 배분하며,
다음 배치를 어떻게 선택하는지**를 설명한다. 세부 정의와 구현 설정은
[Appendix 초안](mcts-algorithm-appendix.md)에 정리했다.
현재 구현 확인일은 2026-10-21이다.

## 1. 본문 초안

We formulate stone stacking as a sequential placement problem. A state
describes the current assembly and the remaining stones, and an action selects
a stone and its placement pose. We use simulation-guided Monte Carlo tree
search (MCTS) to evaluate both the immediate placement and its potential for
subsequent construction.

**Geometry-informed action proposals.** At each state, we generate placement
hypotheses using object geometry and the available support region. A local
placement optimizer refines these hypotheses before they enter the search
tree. At the root, an elite-guided population procedure combines pose
perturbations around promising hypotheses with fresh proposals, providing
local refinement and continued exploration of alternative placements.

**Batched tree search.** We select branches using PUCT, combining accumulated
returns with geometric and equilibrium-based priors. Progressive widening
controls descendant branching as evidence accumulates. Temporary visit
reservations distribute candidate evaluations within a batch, allowing multiple
hypotheses to undergo batched pose optimization and physical simulation.
Transition rewards combine construction progress and geometric quality with
equilibrium and disturbance robustness. Rewards are backed up through the
tree, with a finite-horizon value estimate at nonterminal leaves.

The reward formulation is described in [Reward Design](reward-design.md).

**Physical evaluation.** Each transition first settles the new stone against
the fixed assembly, then evaluates the resulting structure with eligible
supporting stones free to move. Ordinary tree expansions use shorter
simulations, while root placements and selected continuations receive longer
validation. For nominally settled and physically admissible results, additional
trials perturb the new stone's initial velocity and measure the worst peak
motion across dynamic stones. These trials contribute a robustness penalty
and can reject placements that fail the recovery criteria.

**Continuation-aware action selection.** We combine the backed-up mean return
of a root action with the discounted return of a physically validated
continuation:

$$
a^*=\arg\max_{a\in\mathcal A_{\mathrm{val}}(s)}
\left[(1-\lambda)\overline Q(s,a)
+\lambda G_{\mathrm{val}}(s,a)\right].
$$

Here $\mathcal A_{\mathrm{val}}(s)$ contains physically validated root actions
with an available validated continuation, $\overline Q$ is the mean backed-up
return, and $G_{\mathrm{val}}$ is the discounted return along a validated path,
including the immediate placement. Bounded repair and proposal retries seek
missing continuations. If no qualifying continuation is found within the
budget, an explicit fallback can select an otherwise physically validated
root action by its mean return. We execute only the next placement and reuse
its subtree after state update and revalidation.

## 2. Appendix 구성안

| Appendix 항목 | 포함할 상세 정의 |
|---|---|
| A. Problem and objective | State/action, [reward 항과 정규화](reward-design-appendix.md), 실패 판정 |
| B. Placement proposals | Orientation bank, support/area score, 샘플링 분포, root population refinement |
| C. Tree search | PUCT prior, progressive widening, batch reservation, leaf estimate와 실제 backup discount |
| D. Physical evaluation | Fixed-scene settling, SHORT/LONG, 고정 물체, 외란 분포와 excursion/recovery 기준 |
| E. Validation and recovery | Continuation return, 경로 검증 순서, repair·retry·fallback, subtree 재사용 |
| F. Experimental settings | 플랫폼별 가중치·시간·탐색 budget·backend·seed와 실제 사용 설정 |

상세 자료는 [MCTS 정의](mcts-algorithm-appendix.md),
[초기 자세 샘플링](initial-pose-sampling.md),
[시뮬레이션](mcts-simulation.md)에 있다. 본문의 $G_{\mathrm{val}}$은 발견해 검증한
경로의 return이며, 가능한 모든 후속 경로에 대한 전역 최댓값을 뜻하지 않는다.
