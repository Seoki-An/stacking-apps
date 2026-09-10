"""Load a committed placement for simulation using its recorded assets."""
from stacking_common.artifacts import StackRunStore
from stacking_common.artifacts.execution import build_execution_command
from stacking_excavator.planning.replay import recorded_planning_profile, verify_replay_assets


def load_placement(repository, path, *, step=None):
    if StackRunStore.info(path).metadata.get('platform') != 'excavator':
        raise ValueError('simulation requires an excavator run')
    plans = [p for p in StackRunStore.load_committed_placement_plans(path)
             if p.result.succeeded and (step is None or p.step == step)]
    if not plans:
        raise ValueError('run has no committed successful placement for this step')
    plan = plans[-1]
    contexts = {c.sha256: c for c in StackRunStore.load_run_contexts(path)}
    context = contexts[plan.context_sha256]
    if context.profiles.get('platform') != 'excavator':
        raise ValueError('recorded context is not excavator')
    verify_replay_assets(repository, context, recorded_planning_profile(context))
    return plan, context


def load_command(repository, path, *, step=None):
    plan, context = load_placement(repository, path, step=step)
    return build_execution_command(plan, context)
