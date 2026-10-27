# 원격 코드 프로젝트의 Codex에 전달할 요청

첨부/복사한 `mcts-validation-and-benchmark-20261027.md`를 읽고 현재 실행 코드와 정렬해줘. 문서는 논문 프로젝트에서 정리한 가설과 reference이며, 최신 코드가 이미 확인됐다는 뜻은 아니야.

1. 실제 실행 commit/profile/runtime overrides를 확인하고, SHORT/LONG 평가, root/continuation validation, G_val 집계, repair/fallback, progressive widening의 현재 동작을 코드 근거와 함께 정리해줘. 특히 초기 first-passing continuation과 repair의 max-return 처리가 실제로 다른지 확인해줘.
2. `mean` vs `mixed`가 ranking 외에 validation/repair까지 바꾸는지 확인하고, G_val의 효과만 비교할 수 있도록 필요한 최소 설정 분리와 로그를 준비해줘. 기존 기본 동작은 보존하고, 의미가 바뀌는 수정은 이유와 전후 차이를 명시해줘.
3. 작은 테스트로 G_val 할인합, parent-state 일관성, validation cache, state drift, 실패와 fallback을 확인해줘. SHORT에서 거절된 후보가 LONG에서 다시 검증되지 않아 발생할 수 있는 선택 편향도 검토해줘.
4. 동일 후보 SHORT/LONG 진단, 동일 검증 후보 집합에서 Q-only/mixed reranking, 전체 시간 budget을 맞춘 stacking 비교 순서로 benchmark 계획을 작성해줘. 속도와 feasibility 개선을 아직 입증된 결과로 취급하지 마.
5. 실제 존재하는 CLI/config로 검증된 smoke/pilot 실행 명령과 결과 schema를 제시하고, 가능한 작은 smoke test까지만 실행해줘. 대규모 sweep이나 실제 로봇 실행은 시작하지 말고 비용과 계획을 먼저 정리해줘.

완료 후 논문 프로젝트에 다시 가져갈 `alignment-and-benchmark-report.md`를 작성해줘. 코드에서 확인된 사실, 수정 사항, 테스트 결과, 남은 결정, 논문에 쓸 수 있는 설명을 구분해줘. 특히 **탐색 중 저비용 simulation + 실행 전 경로 검증 + 검증 경로 return을 이용한 최종 선택**의 세 역할을 정확히 설명할 수 있어야 해.
