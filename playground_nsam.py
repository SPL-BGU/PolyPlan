import os
import sys
import shutil

import time
from stable_baselines3.common.evaluation import evaluate_policy

from envs import (
    BasicMinecraft,
    IntermediateMinecraft,
    AdvancedMinecraft,
    MaskedMinecraft,
)

from agents import ExploringSam
from agents import FixedScriptAgent

from planning import validator


import numpy as np
import pandas as pd

SEED = 63
np.random.seed(SEED)  # random seed for reproducibility

# import logging

# logging.root.setLevel(logging.INFO)

from enhsp import ENHSP
from metric_ff import MetricFF

from pathlib import Path


def evaluate(env, plan, test_set, map_size):
    """Evaluate the trained model"""
    avg = []
    output_directory_path = f"{os.getcwd()}/dataset/{map_size}"
    for problem_index in test_set:
        domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
        env.set_domain(domain_path)
        env.reset()
        model = FixedScriptAgent(env, script=plan)

        rewards, _ = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=1,
            return_episode_rewards=True,
            deterministic=True,
        )
        avg.append(sum(rewards) / len(rewards))

    # print("Average Reward:", avg)
    # print(f"{id}.Total Average Reward:", sum(avg) / len(avg))
    return sum(avg) / len(avg)


def main(map_type, map_size, planner, use_fluents_map):
    if map_type == "basic":
        env_index = 0  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft, 3: MaskedMinecraft
    elif map_type == "advanced":
        env_index = 2  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft, 3: MaskedMinecraft

    minecraft = [
        BasicMinecraft,
        IntermediateMinecraft,
        AdvancedMinecraft,
        MaskedMinecraft,
    ][env_index]

    if env_index == 0:
        p_domain = "planning/basic_minecraft_domain.pddl"
        p_problem = "planning/basic_minecraft_problem.pddl"
        fluents_map = "planning/basic_minecraft_fluents_map.json"
    elif env_index == 1:
        p_domain = "planning/intermediate_minecraft_domain.pddl"
        p_problem = "planning/intermediate_minecraft_problem.pddl"
        fluents_map = "planning/intermediate_minecraft_fluents_map.json"
    else:
        p_domain = "planning/advanced_minecraft_domain.pddl"
        p_problem = "planning/advanced_minecraft_problem.pddl"
        fluents_map = "planning/advanced_minecraft_fluents_map.json"

    if map_type == "basic":
        env = minecraft(visually=False, start_pal=True, keep_alive=False)
    elif map_type == "advanced":
        env = minecraft(
            visually=False, start_pal=True, keep_alive=False, map_size=map_size
        )

    map_size = f"{map_size}X{map_size}"
    chunk_size = 160
    timeout = 5

    df = pd.read_csv("kfolds.csv")
    for index, row in df.iterrows():
        train_idx = eval(row["train_idx"])
        val_idx = eval(row["val_idx"])

        exploring_sam = ExploringSam(
            env,
            domain=p_domain,
            problem=p_problem,
            fluents_map=fluents_map,
            save_interval=1,
            output_dir="solutions",
        )
        logdir = f"logs/exploring_sam"
        dir_index = 1
        while os.path.exists(f"{logdir}/{dir_index}") and len(
            os.listdir(f"{logdir}/{dir_index}")
        ):
            dir_index += 1
        logdir = f"{logdir}/{dir_index}"
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        file = open(f"{logdir}/results.txt", "w", encoding="utf-8")

        if planner == "ENHSP":
            planner = ENHSP()
        else:
            planner = MetricFF()

        # run on 1 problem
        for start_index in [train_idx[0]]:
            shutil.copy(
                f"{os.getcwd()}/dataset/{map_size}/{map_type}_map_instance_{start_index}.pddl",
                f"{os.getcwd()}/solutions/{map_type}_map_instance_{start_index}.pddl",
            )
            shutil.copy(
                f"{os.getcwd()}/dataset/{map_size}/{map_type}_map_instance_{start_index}.solution",
                f"{os.getcwd()}/solutions/{map_type}_map_instance_{start_index}.solution",
            )

            exploring_sam.active_nsam(
                parser_trajectories=True, use_fluents_map=use_fluents_map
            )

            t1 = time.time()
            total = 0
            to = 0
            unsol = 0
            unval = 0
            plan_tl = 0

            for problem_index in val_idx:
                domain = f"{os.getcwd()}/solutions/domain1.pddl"
                problem = f"{os.getcwd()}/dataset/{map_size}/{map_type}_map_instance_{problem_index}.pddl"

                plan = planner.create_plan(domain, problem, timeout=timeout)

                if len(plan) != 0:
                    tloc = os.getcwd() + "/solutions/temp_plan.txt"
                    odomain = os.getcwd() + "/solutions/domain.pddl"
                    with open(tloc, "w") as tfile:
                        tfile.write("\n".join(plan))

                    valid = validator(Path(odomain), Path(problem), Path(tloc))

                    if valid:
                        if len(plan) > env.max_rounds:
                            plan_tl += 1
                        else:
                            random_check = np.random.randint(1, 11)
                            if random_check == 1:
                                if (
                                    evaluate(env, plan, [problem_index], map_size) > 0
                                ) != valid:
                                    raise Exception(
                                        f"result in simulator and validator are not equal, problem {problem_index}"
                                    )

                            total += 1
                    else:
                        unval += 1
                elif planner.error_flag == 2:
                    to += 1
                else:
                    unsol += 1
            t2 = time.time()
            print(f"Time to evaluate: {t2 - t1}")
            file.write(
                f"solved: {total/len(val_idx)}, not valid: {unval/len(val_idx)}, timeout: {to/len(val_idx)}, plan too long: {plan_tl/len(val_idx)}, unsolvable: {unsol/len(val_idx)}\n"
            )

        # run on all the problems
        for problem_start_index in range(0, len(train_idx), chunk_size):
            for problem_index in train_idx[
                problem_start_index : problem_start_index + chunk_size
            ]:
                shutil.copy(
                    f"{os.getcwd()}/dataset/{map_size}/{map_type}_map_instance_{problem_index}.pddl",
                    f"{os.getcwd()}/solutions/{map_type}_map_instance_{problem_index}.pddl",
                )
                shutil.copy(
                    f"{os.getcwd()}/dataset/{map_size}/{map_type}_map_instance_{problem_index}.solution",
                    f"{os.getcwd()}/solutions/{map_type}_map_instance_{problem_index}.solution",
                )

            exploring_sam.active_nsam(
                parser_trajectories=True, use_fluents_map=use_fluents_map
            )

            t1 = time.time()
            total = 0
            to = 0
            unsol = 0
            unval = 0
            plan_tl = 0

            for problem_index in val_idx:
                domain = f"{os.getcwd()}/solutions/domain{problem_start_index + chunk_size}.pddl"
                # domain = f"/solutions/domain900.pddl"
                problem = f"{os.getcwd()}/dataset/{map_size}/{map_type}_map_instance_{problem_index}.pddl"

                # for solver in [MetricFF]:
                # planner = solver()
                plan = planner.create_plan(domain, problem, timeout=timeout)
                # if len(plan) > 0:
                #     break

                if len(plan) != 0:
                    tloc = os.getcwd() + "/solutions/temp_plan.txt"
                    odomain = os.getcwd() + "/solutions/domain.pddl"
                    with open(tloc, "w") as tfile:
                        tfile.write("\n".join(plan))

                    valid = validator(Path(odomain), Path(problem), Path(tloc))

                    if valid:
                        if len(plan) > env.max_rounds:
                            plan_tl += 1
                        else:
                            random_check = np.random.randint(1, 11)
                            if random_check == 1:
                                if (
                                    evaluate(env, plan, [problem_index], map_size) > 0
                                ) != valid:
                                    raise Exception(
                                        f"result in simulator and validator are not equal, problem {problem_index}"
                                    )

                            total += 1
                    else:
                        unval += 1
                elif planner.error_flag == 2:
                    to += 1
                else:
                    unsol += 1
            t2 = time.time()
            print(f"Time to evaluate: {t2 - t1}")
            file.write(
                f"solved: {total/len(val_idx)}, not valid: {unval/len(val_idx)}, timeout: {to/len(val_idx)}, plan too long: {plan_tl/len(val_idx)}, unsolvable: {unsol/len(val_idx)}\n"
            )

        file.close()

        os.rename(
            "/home/benjamin/Projects/PolyPlan/solutions",
            f"/home/benjamin/Projects/PolyPlan/solutions{dir_index}",
        )
        os.makedirs("/home/benjamin/Projects/PolyPlan/solutions")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        map_type = sys.argv[1]
        map_size = int(sys.argv[2])
        planner = sys.argv[3]
        use_fluents_map = sys.argv[4] == "True"
        main(map_type, map_size, planner, use_fluents_map)
    else:
        print("Please provide a variable as a command-line argument.")
        print(
            "Example: python playground_offline.py [basic/advanced] [6/10] [ENHSP/FF] [True/False]"
        )
        # main("basic", 6, "ENHSP", True)
