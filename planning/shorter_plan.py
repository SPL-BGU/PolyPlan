import sys
import numpy as np
import config as CONFIG
import copy

sys.path.append(CONFIG.NSAM_PATH)

from pddl_plus_parser.lisp_parsers import ProblemParser
from pddl_plus_parser.models import (
    Operator,
    ActionCall,
    ObservedComponent,
    CompoundPrecondition,
)

from utils import AdvancedActionsDecoder


def parse_action_call(action_call: str) -> ActionCall:
    """Parses the string representing the action call in the plan sequence.

    :param action_call: the string representing the action call.
    :return: the object representing the action name and its parameters.
    """
    action_data = action_call.lower().replace("(", " ( ").replace(")", " ) ").split()
    action_data = action_data[1:-1]
    return ActionCall(name=action_data[0], grounded_parameters=action_data[1:])


def shorter_plan(env, observation, problem_file_path, domain, learned_model):
    uproblem = ProblemParser(problem_path=problem_file_path, domain=domain)
    problem = uproblem.parse_problem()

    decoder = AdvancedActionsDecoder(env.map_size**2)
    decoder.agent_state = {
        "position": np.zeros(
            (1,),
            dtype=np.int16,
        )
    }
    decoder.crafting_table_cell = env.decoder.crafting_table_cell

    tdomain = copy.deepcopy(learned_model)
    for action in tdomain.actions:
        tdomain.actions[action].preconditions = CompoundPrecondition()

    shorter_plan = []
    last_seen = []
    size = 0
    for component in observation.components:
        if component.previous_state == component.next_state:
            continue

        state = component.previous_state.serialize()
        index = last_seen.index(state) if state in last_seen else -1

        if index != -1:
            if index == 0:
                shorter_plan = []
                last_seen = []
                size = 0
            else:
                shorter_plan = shorter_plan[:index]
                last_seen = last_seen[:index]  # remove also the index from last_seen
                size = index - 1

        shorter_plan.append(component)
        last_seen.append(state)
        size += 1

    j = len(shorter_plan) - 1
    while j > 0:
        from_state = shorter_plan[j - 1].previous_state
        to_state = shorter_plan[j].next_state

        pos = [cell for cell in from_state.state_predicates["(position ?c)"]][
            0
        ].grounded_objects[0]
        if pos == "crafting_table":
            pos = decoder.crafting_table_cell
        else:
            pos = int(pos.replace("cell", ""))
        decoder.agent_state["position"][0] = pos

        for action in range(decoder.get_actions_size()):
            action = decoder.decode_to_planning(action)
            action = f"({action})"

            action_descriptor = parse_action_call(action)
            try:
                operator = Operator(
                    action=tdomain.actions[action_descriptor.name],
                    domain=tdomain,
                    grounded_action_call=action_descriptor.parameters,
                    problem_objects=problem.objects,
                )
            except KeyError:
                continue

            # check if the affect of the operator on from_state will result in to_state
            next_state = operator.apply(from_state, skip_validation=True)

            if next_state == to_state:
                operator = Operator(
                    action=learned_model.actions[action_descriptor.name],
                    domain=learned_model,
                    grounded_action_call=action_descriptor.parameters,
                    problem_objects=problem.objects,
                )

                # check if the operator is applicable
                try:
                    next_state = operator.apply(from_state)
                except ValueError:
                    continue

                shorter_plan.pop(j)

                shorter_plan[j - 1] = ObservedComponent(
                    from_state, action_descriptor, to_state
                )
                break
        j -= 1

    observation.components = shorter_plan
    return observation
