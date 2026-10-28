# Reward Design: Appendix 상세 정의 초안

> Implementation update (2026-10-28): the surface term now rewards the reduction
> in normalized relief, `weight × (roughness_before − roughness_after)`, rather
> than penalizing absolute relief. The dated draft below describes the earlier
> rule; see [the current specification](initial-pose-sampling.md#mcts-surface-flatness-reward).

[논문 본문 초안](reward-design.md)의 feature를 현재 코드와 대응시킨다.
**확인일: 2026-10-21.** 본문의 $\phi$는 보상 feature, $\psi$는 비음수 penalty이며,
아래 값은 excavator/tabletop repository 기본값이다.

## A. 기호와 최종 reward

$i$는 새로 놓은 돌, $T_i$는 nominal simulation 후 자세, $\mathcal T$는 target
volume이다. 돌의 부피 $V_i$는 convex-component volume의 합이며, 초기 inventory에
대해 고정된 기준 부피와 길이를

$$
\bar V=\frac1N\sum_{j=1}^{N}V_j,\qquad L_{\mathrm{ref}}=\bar V^{1/3}
$$

로 정의한다. 남은 돌이나 현재 놓인 돌만으로 이 기준을 다시 계산하지 않는다.

$$
r=w_p\phi_p+w_c\phi_c+w_o\phi_o
-w_e\psi_e-w_f\psi_f-w_r\psi_r.
$$

| 본문 기호 | Configuration weight | Excavator | Tabletop |
|---|---|---:|---:|
| $w_p$ | `target_overlap_increment` | 1 | 1 |
| $w_c$ | `object_containment` | 2 | 2 |
| Tabletop 추가항 $w_o$ | `inward_orientation` | 0 | 2 |
| $w_e$ | `stability` | 2 | 2 |
| $w_f$ | `surface_flatness` | 2 | 2 |
| $w_r$ | `place_stability` | 2 | 2 |
| 현재 본문에서 생략 | `large_object_lower` | 0 | 0 |

코드는 penalty feature를 음수로 저장하고 양의 가중치를 곱한다. 본문은 같은 값을
비음수 $\psi$에 대한 뺄셈으로 표현한다. 전체 reward를 하나의 제한된 구간으로
clipping하지는 않는다.

현재 excavator는 $w_o=0$이므로 본문에서는 orientation 항을 생략한다. Tabletop은
$w_o=2$를 유지한다. 두 플랫폼 모두 `large_object_lower`는 비활성이다.

본문의 $r(s,a,s')$는 transition simulation에서 계산·기록한 outcome을 포함해
평가한다. Peak excursion처럼 trajectory에 의존하는 값을 최종 pose만으로
복원할 수는 없다. 구현에서는 action diagnostics와 transition 결과에 이를 저장한다.

## B. Construction progress와 containment

돌의 local convex-component union을 $\mathcal V_i$라 할 때 연속적인 교집합 부피는

$$
V_{i\cap T}=\int_{\mathcal T}
\mathbf 1[T_i^{-1}p\in\mathcal V_i]\,d^3p.
$$

실제로는 target 안의 voxel center $\mathcal P_T$에서

$$
\widehat V_{i\cap T}=\delta_v^3\sum_{p\in\mathcal P_T}
\mathbf 1[T_i^{-1}p\in\mathcal V_i],\qquad
C_i=\operatorname{clip}\left(\frac{\widehat V_{i\cap T}}{V_i},0,1\right)
$$

로 근사한다. Current voxel spacing은 excavator 0.03 m, tabletop 0.003 m다.
여러 component에 속한 voxel은 한 번만 세지만 $V_i$에는 component volume이 각각
더해진다. Component overlap이 있으면 분자와 분모의 부피 정의가 완전히 같지는 않다.

본문의 두 feature는

$$
\phi_p=\frac{C_iV_i}{\bar V},\qquad
\phi_c=\operatorname{clip}(C_i-\tau_c,0,1),\qquad\tau_c=0.5
$$

이다. 완전히 포함된 평균 크기 돌은 $\phi_p=1$, $\phi_c=0.5$를 얻는다.
현재 가중치에서는 두 항이 각각 reward 1을 기여한다. 큰 돌의 $\phi_p$는 1보다
클 수 있으며, 포함 비율과 포함 부피를 서로 다른 feature로 사용한다.

코드의 기존 increment는 $\Delta I_T=C_iV_i/V_T$이고, 여기에 $V_T/\bar V$를 곱해
위의 $\phi_p$를 만든다. `target_IoU` 계열 diagnostic은 개별 돌의 target overlap을
합산한 값이다. 이 값은 구조물 union과 target 사이의 통상적인 IoU가 아니다.
또한 progress는 기존 돌의 재배치에 따른 coverage 증감까지 포함한
$\Phi(s')-\Phi(s)$ 대신 새 돌의 contained volume에 해당한다.

## C. Equilibrium penalty

Native posegen은 candidate와 placed bodies의 평균 질량으로 질량을 정규화한다.
Adapter는 반환된 net wrench에 그 평균 질량을 곱해 force와 moment를 복원한다.
복원된 residual force $f_b$와 moment $\tau_b$에 대해

$$
E=\max_b\left[
\frac{\|f_b\|_2}{m_bg}
+\frac{\|\tau_b\|_2}{m_bgV_b^{1/3}}
\right],\qquad
\psi_e=\operatorname{clip}(E,0,1).
$$

$b$는 native result에 net wrench가 반환된 body다. $m_b,V_b$는 그 body의 질량과
component-volume sum이며, $g$는 posegen profile의 gravity다. Diagnostic 이름은
`posegen_equilibrium_residual_normalized`다.

이 값은 **posegen 결과에서 계산된 residual**이다. Structural simulation의 최종
장면에서 equilibrium force solve를 새로 수행해 얻은 feature는 아니다.

Diagnostic이 없는 historical action은
$\psi_e=\operatorname{clip}(c_{\mathrm{feq}}/\tau_e,0,1)$를 사용한다.
현재 $\tau_e=1$이다. Legacy feasibility filter의 $c_{\mathrm{feq}}$ threshold와
위 dimensionless reward feature는 별도로 해석해야 한다.

## D. Upper-surface relief penalty

Nominal 최종 장면의 component meshes에서 target의 nominal XY rectangle 위
32 × 32 cell-center grid로 upper envelope를 rasterize한다. 높이는 target base에
대한 상대값이며 빈 칸은 0이다. Target 높이를 넘는 값은 위쪽에서 자르지 않는다.

$$
\psi_f=\frac{\max_j h_j-\frac1M\sum_{j=1}^{M}h_j}{L_{\mathrm{ref}}},
\qquad M=32^2.
$$

이 항은 전체 구조물의 현재 surface relief에 대한 penalty다. 이전 상태와의
relief 차이를 사용하지 않는다. 최대값과 평균값이 같으면 표면 분포가 달라도 같은
점수를 받으며, 돌 아래 가려진 cavity는 검출하지 못한다.

본문의 $h_{\max}=\max_jh_j$, $\overline h=M^{-1}\sum_jh_j$는 이 grid 측정값이다.
Flatness feature는 높이 분산이 아니라 max-minus-mean relief이며, weighted reward
기여는 $-w_f\psi_f$다.

## E. Upper/inward orientation reward: tabletop에서 활성

새 돌의 component-face normal을 nominal 최종 자세로 회전시킨다. 이 중
$n_{\ell,z}\ge0$인 face normal을 면적 $A_\ell$로 가중 평균한 뒤 정규화한다.

$$
n=\frac{\sum_{\ell:n_{\ell,z}\ge0}A_\ell n_\ell}
{\left\|\sum_{\ell:n_{\ell,z}\ge0}A_\ell n_\ell\right\|_2},
\qquad
u=\operatorname{clip}\left(\frac{n_z-0.7}{0.3},0,1\right).
$$

면적 가중치 합이 수치적으로 0이면 uniform weight를 사용한다. 유효한 normal을
얻지 못하면 orientation feature는 0이다. 이 대표 normal은 component geometry의
근사이며, 실제 외부에 노출된 면을 visibility 검사로 식별하는 것은 아니다.

$d=o_{xy}-p_{i,xy}$를 새 돌의 origin에서 target 중심으로 향하는 방향으로 두고,
$v=n_{xy}$라 하자. 안쪽 정렬 점수는

$$
\iota=
\begin{cases}
0.5,&\|v\|_2<0.05\ \text{or}\ \|d\|_2<10^{-9}\,\mathrm m,\\
\operatorname{clip}\left(
\dfrac{\frac{v^\mathsf T d}{\|v\|_2\|d\|_2}-\cos40^\circ}
{1-\cos40^\circ},0,1\right),&\text{otherwise}.
\end{cases}
\qquad \phi_o=u\,\iota.
$$

따라서 거의 수직인 대표 normal이나 target 중심의 돌은 방향 항에서 중립값 0.5를
받는다. $\phi_o$의 범위는 $[0,1]$이며 실제 안정성 판정은 별도의 물리 평가가 맡는다.

## F. Disturbance-robustness penalty

Trial $k$는 nominal structural snapshot에서 시작하며 새 돌에만 초기 선속도와
각속도 외란을 준다. Support를 포함한 dynamic body들의 최대 움직임을 측정한다.

$$
D=\max_{k,b,t}\left[
\|p_b^{(k)}(t)-p_b^0\|_2+
\lambda_{\mathrm{rot}}\theta(q_b^{(k)}(t),q_b^0)\right],
\qquad \psi_r=\frac{\overline D}{D_{\mathrm{ref}}}.
$$

$\theta$는 quaternion의 최단 회전각(rad)이다. $D$는 단순 final displacement가
아닌 **peak excursion**이다. $\overline D$는 reward 계산을 위해 nonfinite $D$를
지정 값으로 대체하고 `max_displacement`로 위쪽을 자른 값이다. 원래 $D$와 trial
실패 사유를 이용한 rejection은 이 clipping으로 취소되지 않는다.

정규화된 quaternion이면 $\theta(q,q^0)=2\arccos(|q^\mathsf Tq^0|)$다.
수치 계산에서는 inner product를 $[0,1]$로 clipping한다. 처리 순서를 명시하면

$$
D_{\mathrm{finite}}=
\begin{cases}
D,&D\text{가 finite인 경우},\\
D_{\mathrm{nf}},&\text{otherwise},
\end{cases}
\qquad
\overline D=\min(D_{\mathrm{finite}},D_{\mathrm{clip}}),
\qquad r_{\mathrm{robust}}=-w_r\psi_r.
$$

현재 두 플랫폼 모두 reward clipping 상한을 지정한다. 이 처리와 별개로 nonfinite
결과는 물리 평가에서 실패이며, clipping된 유한 penalty만으로 통과시킬 수 없다.

| 항목 | Excavator | Tabletop |
|---|---:|---:|
| Trial 수 | 4 | 4 |
| Linear velocity std, 각 world 축 | 0.05 m/s | 0.02 m/s |
| Angular velocity std, 각 body 축 | 5°/s | 5°/s |
| $\lambda_{\mathrm{rot}}$ | 0.1 m/rad | 0.01 m/rad |
| $D_{\mathrm{ref}}$ | 0.15 m | 0.015 m |
| Excursion 실패 한도 | 0.15 m | 0.015 m |
| Nonfinite 대체값 | 2.0 m | 0.2 m |
| Reward용 clipping 상한 | 1.0 m | 0.1 m |

Nominal 결과가 settled이고 backend의 물리 조건을 통과한 경우에 trial을 실행한다.
실행하지 않은 trial set의 backend 반환 displacement는 0일 수 있으므로,
`place_robustness_evaluated`를 함께 확인해야 한다. 0-valued feature만으로 robustness
검증에 성공했다고 해석하지 않는다. 자세한 recovery 조건과 SHORT/LONG 적용은
[시뮬레이션 문서](mcts-simulation.md)에 있다.

## G. 비활성 항과 hard rejection

`large_object_lower`의 feature는 inventory 내 정규화 부피와 낮은 높이 선호의 곱이다.
부피가 모두 같으면 정규화 부피를 0.5로 두고, 아니면
$v_i=(V_i-V_{\min})/(V_{\max}-V_{\min})$로 둔다. 높이 선호는
$1-\operatorname{clip}((p_{i,z}-o_z)/\max(h_T,10^{-6}\,\mathrm m),0,1)$이다.
현재 가중치가 0이므로 본문 목적함수에서는 생략한다. Support count와 ground
contact도 diagnostic으로 기록되지만 별도의 weighted reward 항은 아니다.

Transition은 simulation failure, 미안정화, 낮은 containment 및 적용되는 floor 조건에
의해 탐색에서 제외될 수 있다. 큰 음의 utility를 주는 것과 이 admission 판정은
별개다. 실패 node의 backup은 환경 reward 대신 `failure_reward=-1`을 사용하며,
실패 node 자신은 0-valued visit을 기록하고 ancestor에는 실패 return을 전달한다.

XY proposal score, root CEM ranking, PUCT prior도 본문의 transition reward와
구분해야 한다. 현재 prior는 equilibrium/containment를 사용하고 CEM은 여기에
정규화한 proposal score를 더한다. 상세 흐름은
[MCTS appendix](mcts-algorithm-appendix.md)에 정리되어 있다.

## H. 코드 근거

| 항목 | 근거 |
|---|---|
| Feature와 가중합 | [reward.py](../packages/stacking-common/src/stacking_common/environment/reward.py): `StackingReward.compute` |
| 대표 normal | [_geometry.py](../packages/stacking-common/src/stacking_common/environment/_geometry.py): `exposed_surface_normal`; [object.py](../packages/stacking-common/src/stacking_common/domain/object.py): `representative_face_normals` |
| 부피와 containment | [object.py](../packages/stacking-common/src/stacking_common/domain/object.py): `volume`; [overlap.py](../packages/stacking-common/src/stacking_common/environment/overlap.py): `VoxelOverlapEvaluator` |
| Surface grid | [surface.py](../packages/stacking-common/src/stacking_common/environment/surface.py): `SurfaceFlatness` |
| Physical wrench와 equilibrium | [core_solver.py](../packages/stacking-common/src/stacking_common/actions/core_solver.py): `_placement_values` |
| 외란과 excursion | [robustness.py](../packages/stacking-common/src/stacking_common/environment/robustness.py): `trial_outcome`, `scene_trial_outcome` |
| Transition 거절 | [adapters.py](../packages/stacking-common/src/stacking_common/search/adapters.py): `_transition_rejection_reason` |
| 실패 backup | [backup.py](../packages/stacking-common/src/stacking_common/search/mcts/backup.py) |
| Platform profile | [Excavator](../apps/excavator/config/environment.yml), [Tabletop](../apps/tabletop/config/environment.yml) |

정규화와 가중치는 현재 구현을 재현하기 위한 정의다. 각 항이 적층 성공률을
높인다는 주장은 별도의 ablation과 실제 평가로 확인해야 한다.

이번 확인에서는 기존 `test_physical_reward_normalization.py`,
`test_surface_flatness.py`, `test_voxel_overlap.py`의 **38개 테스트가 통과**했다.
이는 정규화·surface feature·voxel overlap 구현에 대한 확인이며, reward design의
실제 적층 성능에 대한 검증 결과는 아니다.
