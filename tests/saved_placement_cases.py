"""Load saved candidate actions and their pre-placement scene for replay."""
import json
import sqlite3
import numpy as np
from stacking_common.artifacts import StackRunStore
from stacking_common.domain import Action, State

def load_case(path, step, ids):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
        row = connection.execute("select payload_json from records where "
                                 "kind='mcts_step' and ordinal=?", (step,)).fetchone()
        if row is None:
            raise ValueError(f'no saved step {step}')
        data = json.loads(row[0])
    if step == 1:
        state = State(ids, [], {}, [], [])
    else:
        saved = StackRunStore.load_planning_checkpoint(path, step - 1).state
        state = State(saved.object_ids, saved.placed_object_indices,
                      saved.object_poses, [], [],
                      pose_identified_object_ids=saved.pose_identified_object_ids)
    candidates = [c for c in data['candidates'] if c.get('action_pose') is not None]
    preferred = 'step-1-candidate-9' if step == 1 else data['selected_candidate_id']
    candidates.sort(key=lambda c: c['candidate_id'] != preferred)
    actions = [Action(c['object_index'], np.array(c['action_pose']),
                      np.array(c['init_pose']), object_id=c['object_id']) for c in candidates]
    return state, candidates, actions

