from pathlib import Path
from tqdm import tqdm

from envs import BasicMinecraft, AdvancedMinecraft
from agents import FixedScriptAgent

from planning.metric_ff import MetricFF
from planning.enhsp import ENHSP

from utils import ProblemGenerator
import json


def generate_solutions_basic(output_directory_path: Path, problem_index: int):
    domain = "/home/benjamin/Projects/PolyPlan/planning/basic_minecraft_domain.pddl"
    problem = f"{output_directory_path}/basic_map_instance_{problem_index}.pddl"

    planner = ENHSP()
    plan = planner.create_plan(domain, problem)

    if len(plan) == 0:
        return False

    filename = f"{output_directory_path}/basic_map_instance_{problem_index}.solution"
    with open(filename, "w") as file:
        for item in plan:
            file.write(str(item) + "\n")
    return True


def generate_solutions_advanced(output_directory_path: Path, problem_index: int):
    domain = "/home/benjamin/Projects/PolyPlan/planning/advanced_minecraft_domain.pddl"
    problem = f"{output_directory_path}/advanced_map_instance_{problem_index}.pddl"

    planner = ENHSP()
    plan = planner.create_plan(domain, problem)

    if len(plan) == 0:
        return False

    filename = f"{output_directory_path}/advanced_map_instance_{problem_index}.solution"
    with open(filename, "w") as file:
        for item in plan:
            file.write(str(item) + "\n")
    return True


def generate_solutions_advanced_via_basic(
    output_directory_path: Path, problem_index: int
):
    if not generate_solutions_basic(output_directory_path, problem_index):
        return False

    map_path = f"{output_directory_path}/map_instance_{problem_index}.json"
    with open(map_path, "r") as map_file:
        map_json = json.load(map_file)
    block_list = map_json["features"][2]["blockList"]

    map_size = map_json["features"][1]["pos2"][0] - 1
    transfer_location = lambda x: (int(x[0]) - 1) + ((int(x[2]) - 1) * map_size)

    tree_locations_in_map = []
    for block in block_list:
        if block["blockName"] == "tree":
            tree_locations_in_map.append(transfer_location(block["blockPos"]))

    file = open(
        f"{output_directory_path}/basic_map_instance_{problem_index}.solution", "r"
    )
    basic = file.read().split("\n")
    advanced = f"{output_directory_path}/advanced_map_instance_{problem_index}.solution"
    current_pos = f'cell{transfer_location(map_json["features"][0]["pos"])}'
    with open(advanced, "w") as file:
        for action in basic:
            if action == "(get_log)":
                tree_location = tree_locations_in_map.pop()
                file.write(f"(tp_to {current_pos} cell{tree_location})\n")
                file.write("(break)\n")
                current_pos = f"cell{tree_location}"
            elif action == "(place_tree_tap)":
                tree_location = tree_locations_in_map[0]
                file.write(f"(tp_to {current_pos} cell{tree_location})\n")
                file.write("(place_tree_tap)\n")
                current_pos = f"cell{tree_location}"
            elif action == "(craft_tree_tap)":
                file.write(f"(craft_tree_tap crafting_table)\n")
                current_pos = "crafting_table"
            elif action == "(craft_wooden_pogo)":
                file.write(f"(craft_wooden_pogo crafting_table)\n")
                current_pos = "crafting_table"
            else:
                file.write(action + "\n")

    return True


def valid_problem_basic(output_directory_path: Path, problem_index: int):
    env = BasicMinecraft(start_pal=False, keep_alive=True)

    domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
    env.set_domain(domain_path)

    filename = f"{output_directory_path}/basic_map_instance_{problem_index}.solution"
    fixed_script_agent = FixedScriptAgent(env, filename=filename, human_readable=False)

    env.reset()
    reward = 0
    while not env.done:
        fixed_script_agent.act()
        reward += env.reward
    env.close()

    if reward == 0:
        return False
    return True


def valid_problem_advanced(output_directory_path: Path, problem_index: int):
    map_path = f"{output_directory_path}/map_instance_{problem_index}.json"
    with open(map_path, "r") as map_file:
        map_json = json.load(map_file)
    map_size = map_json["features"][1]["pos2"][0] - 1

    env = AdvancedMinecraft(start_pal=False, keep_alive=True, map_size=map_size)

    domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
    env.set_domain(domain_path)

    filename = f"{output_directory_path}/advanced_map_instance_{problem_index}.solution"
    fixed_script_agent = FixedScriptAgent(env, filename=filename, human_readable=False)

    env.reset()
    reward = 0
    while not env.done:
        fixed_script_agent.act()
        reward += env.reward
    env.close()

    if reward == 0:
        return False
    return True


if __name__ == "__main__":
    output_directory_path = f"/home/benjamin/Projects/PolyPlan/dataset/basic"

    generator = ProblemGenerator(None)
    generator.generate_basic_problems(1000)

    for i in tqdm(range(1000)):
        success = generate_solutions_basic(output_directory_path, i)
        if not success:
            raise Exception(f"No solution found for problem {i} via solver")
        success = valid_problem_basic(output_directory_path, i)
        if not success:
            raise Exception(f"Solution {i} not valid via simulator")
