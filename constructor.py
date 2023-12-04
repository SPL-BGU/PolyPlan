from pathlib import Path
from tqdm import tqdm

from envs import BasicMinecraft, AdvancedMinecraft
from agents import FixedScriptAgent, LearningAgent

from planning.metric_ff import MetricFF
from planning.enhsp import ENHSP

from utils import ProblemGenerator
import json

import pandas as pd

import numpy as np
import random

SEED = 63
np.random.seed(SEED)  # random seed for reproducibility
random.seed(SEED)


def generate_solutions_basic(output_directory_path: Path, problem_index: int) -> bool:
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


def generate_solutions_advanced(
    output_directory_path: Path, problem_index: int
) -> bool:
    domain = "/home/benjamin/Projects/PolyPlan/planning/advanced_minecraft_domain.pddl"
    problem = f"{output_directory_path}/advanced_map_instance_{problem_index}.pddl"

    planner = MetricFF()
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
) -> bool:
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


def valid_problem_basic(output_directory_path: Path, problem_index: int) -> bool:
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


def valid_problem_advanced(output_directory_path: Path, problem_index: int) -> bool:
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


def rule_them_all(output_directory_path: Path, problem_index: int) -> bool:
    env = BasicMinecraft(start_pal=False, keep_alive=True)

    domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
    env.set_domain(domain_path)

    filename = "agents/scripts/macro_actions_script.txt"
    fixed_script_agent = FixedScriptAgent(env, filename=filename, human_readable=True)

    env.reset()
    reward = 0
    for _ in range(fixed_script_agent.length):
        fixed_script_agent.act()
        reward += env.reward
        if env.done:
            break
    env.close()

    if reward == 0:
        return False
    return True


def get_raw_data(map_path: str) -> dict:
    problem_dict = {}

    with open(map_path, "r") as map_file:
        map_json = json.load(map_file)

    # tree cell count
    problem_dict["tree_count"] = len(map_json["features"][2]["blockList"]) - 1

    # starting inventory count
    for item in map_json["features"][5]["itemList"]:
        item = item["itemDef"]
        problem_dict[item["itemName"]] = item["count"]

    # zero items that not start with
    items = [
        "minecraft:log",
        "minecraft:planks",
        "minecraft:stick",
        "polycraft:sack_polyisoprene_pellets",
        "polycraft:tree_tap",
    ]
    for item in items:
        if item not in problem_dict:
            problem_dict[item] = 0

    return problem_dict


if __name__ == "__main__":
    env = BasicMinecraft(visually=False, start_pal=True, keep_alive=True)
    env.close()

    solve_counting = False
    map_size = 6
    num_maps_to_generate = 1000

    output_directory_path = f"dataset/{map_size}X{map_size}"

    # generate problems
    generator = ProblemGenerator("dataset/")
    generator.generate_problems(
        num_maps_to_generate=num_maps_to_generate, map_size=map_size, basic_only=False
    )

    # generate planning solutions for counting map
    if solve_counting:
        for i in tqdm(range(num_maps_to_generate)):  # time taken: 2h
            success = generate_solutions_basic(output_directory_path, i)
            if not success:
                raise Exception(f"No solution found for problem {i} via solver")
            success = valid_problem_basic(output_directory_path, i)
            if not success:
                raise Exception(f"Solution {i} not valid via simulator")

    # generate planning solutions for advanced map
    for i in tqdm(range(num_maps_to_generate)):  # time taken: 2h
        success = generate_solutions_advanced(output_directory_path, i)
        if not success:
            raise Exception(f"No solution found for problem {i} via solver")
        success = valid_problem_advanced(output_directory_path, i)
        if not success:
            raise Exception(f"Solution {i} not valid via simulator")

    map_name = f"{map_size}X{map_size}"

    # generate RL solutions for counting map
    env = BasicMinecraft(visually=False, start_pal=False, keep_alive=True)
    if solve_counting:
        for i in tqdm(range(num_maps_to_generate)):  # time taken: 3h
            domain_path = f"dataset/{map_name}/map_instance_{i}.json"
            env.set_domain(domain_path)
            filename = f"dataset/{map_name}/basic_map_instance_{i}.solution"
            fixed_script_agent = FixedScriptAgent(
                env, filename=filename, human_readable=False
            )

            learning_agent = LearningAgent(env, fixed_script_agent, for_planning=False)

            sol = f"dataset/{map_name}/basic_map_instance_{i}.pkl"
            learning_agent.record_trajectory()
            learning_agent.export_trajectory(sol)
    env.close()

    # generate RL solutions for advanced map
    env = AdvancedMinecraft(
        visually=False, start_pal=False, keep_alive=False, map_size=map_size
    )
    for i in tqdm(range(num_maps_to_generate)):  # time taken: 3h
        domain_path = f"dataset/{map_name}/map_instance_{i}.json"
        env.set_domain(domain_path)
        filename = f"dataset/{map_name}/advanced_map_instance_{i}.solution"
        fixed_script_agent = FixedScriptAgent(
            env, filename=filename, human_readable=False
        )

        learning_agent = LearningAgent(env, fixed_script_agent, for_planning=False)

        sol = f"dataset/{map_name}/advanced_map_instance_{i}.pkl"
        learning_agent.record_trajectory()
        learning_agent.export_trajectory(sol)
    env.close()

    # raw data of the generated maps
    lst = []
    for i in range(num_maps_to_generate):
        map_path = f"{output_directory_path}/map_instance_{i}.json"
        lst.append(get_raw_data(map_path))
    df = pd.DataFrame(lst)
    column_stats = df.agg(["min", "max", "mean", "std"])
    column_stats = column_stats.rename(
        index={
            "min": "Range (Min)",
            "max": "Range (Max)",
            "mean": "Average (Mean)",
            "std": "Standard Deviation",
        }
    )
    duplicate_rows = df[df.duplicated(keep=False)]
    duplicate_count = len(duplicate_rows)
    duplicate_count_df = pd.DataFrame({"Duplicate_Count": [duplicate_count]})
    column_stats = pd.concat([column_stats, duplicate_count_df], ignore_index=True)

    # rule them all count
    # rule_count = 0
    # for i in tqdm(range(num_maps_to_generate)):
    #     success = rule_them_all(output_directory_path, i)
    #     if success:
    #         rule_count += 1
    # print(rule_count)
    # rule_count_df = pd.DataFrame({"Rule_Count": [rule_count]})
    # column_stats = pd.concat([column_stats, rule_count_df], ignore_index=True)

    column_stats.to_csv(f"{output_directory_path}/raw_data.csv", index=True)
