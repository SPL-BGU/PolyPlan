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

from utils import Logger

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


def main(
    map_type,
    map_size_from,
    map_size_to,
    planner,
    use_fluents_map,
    fold,
    timeout,
    max_steps,
):
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
        env = minecraft(
            visually=False, start_pal=True, keep_alive=False, max_steps=max_steps
        )
    elif map_type == "advanced":
        env = minecraft(
            visually=False,
            start_pal=True,
            keep_alive=False,
            max_steps=max_steps,
            map_size=map_size_to,
        )

    map_size_to = f"{map_size_to}X{map_size_to}"
    map_size_from = f"{map_size_from}X{map_size_from}"
    chunk_size = 160

    j = -1
    df = pd.read_csv("kfolds.csv")
    for _, row in df.iterrows():
        # skip to the fold
        j += 1
        if j < fold:
            continue

        train_idx = eval(row["train_idx"])
        val_idx = eval(row["val_idx"])

        # make log directory
        postfix = f"exploring_sam/{map_type}_{map_size_from}_to_{map_size_to}/fluents_map_{use_fluents_map}/timeout_{timeout}/fold_{fold}"
        logdir, models_dir = Logger.create_logdir(postfix)

        file = open(f"{logdir}/results.txt", "w", encoding="utf-8")

        if planner == "FF":
            planner = MetricFF()
        else:
            planner = ENHSP()

        exploring_sam = ExploringSam(
            env,
            domain=p_domain,
            problem=p_problem,
            fluents_map=fluents_map,
            save_interval=1,
            output_dir=models_dir,
        )

        # run on 1 problem
        for start_index in [train_idx[0]]:
            shutil.copy(
                f"{os.getcwd()}/dataset/{map_size_from}/{map_type}_map_instance_{start_index}.pddl",
                f"{os.getcwd()}/{models_dir}/{map_type}_map_instance_{start_index}.pddl",
            )
            shutil.copy(
                f"{os.getcwd()}/dataset/{map_size_from}/{map_type}_map_instance_{start_index}.solution",
                f"{os.getcwd()}/{models_dir}/{map_type}_map_instance_{start_index}.solution",
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
                domain = f"{os.getcwd()}/{models_dir}/domain1.pddl"
                problem = f"{os.getcwd()}/dataset/{map_size_to}/{map_type}_map_instance_{problem_index}.pddl"

                plan = planner.create_plan(domain, problem, timeout=timeout)

                if len(plan) != 0:
                    tloc = f"{os.getcwd()}/{models_dir}/temp_plan.txt"
                    odomain = f"{os.getcwd()}/{models_dir}/domain.pddl"
                    with open(tloc, "w") as tfile:
                        tfile.write("\n".join(plan))

                    valid = validator(Path(odomain), Path(problem), Path(tloc))

                    if valid:
                        if len(plan) > env.max_steps:
                            plan_tl += 1
                        else:
                            random_check = np.random.randint(1, 11)
                            if random_check == 1:
                                if (
                                    evaluate(env, plan, [problem_index], map_size_to)
                                    > 0
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
                    f"{os.getcwd()}/dataset/{map_size_from}/{map_type}_map_instance_{problem_index}.pddl",
                    f"{os.getcwd()}/{models_dir}/{map_type}_map_instance_{problem_index}.pddl",
                )
                shutil.copy(
                    f"{os.getcwd()}/dataset/{map_size_from}/{map_type}_map_instance_{problem_index}.solution",
                    f"{os.getcwd()}/{models_dir}/{map_type}_map_instance_{problem_index}.solution",
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
                domain = f"{os.getcwd()}/{models_dir}/domain{problem_start_index + chunk_size}.pddl"
                problem = f"{os.getcwd()}/dataset/{map_size_to}/{map_type}_map_instance_{problem_index}.pddl"

                plan = planner.create_plan(domain, problem, timeout=timeout)

                if len(plan) != 0:
                    tloc = f"{os.getcwd()}/{models_dir}/temp_plan.txt"
                    odomain = f"{os.getcwd()}/{models_dir}/domain.pddl"
                    with open(tloc, "w") as tfile:
                        tfile.write("\n".join(plan))

                    valid = validator(Path(odomain), Path(problem), Path(tloc))

                    if valid:
                        if len(plan) > env.max_steps:
                            plan_tl += 1
                        else:
                            random_check = np.random.randint(1, 11)
                            if random_check == 1:
                                if (
                                    evaluate(env, plan, [problem_index], map_size_to)
                                    > 0
                                ) != valid:
                                    raise Exception(
                                        f"result in simulator and validator are not equal, problem {problem_index}"
                                    )

                            total += 1
                elif planner.error_flag == 2:
                    to += 1
                else:
                    unsol += 1
            t2 = time.time()
            print(f"Time to evaluate: {t2 - t1}")
            file.write(
                f"solved: {total/len(val_idx)}, not valid: {unval/len(val_idx)}, timeout: {to/len(val_idx)}, plan too long: {plan_tl/len(val_idx)}, unsolvable: {unsol/len(val_idx)}\n"
            )
            file.flush()

        file.close()

        # clean models directory
        files_to_keep = ["domain1.pddl"] + [
            f"domain{i}.pddl" for i in range(chunk_size, 801, chunk_size)
        ]

        for filename in os.listdir(models_dir):
            file_path = os.path.join(models_dir, filename)
            if filename not in files_to_keep:
                os.remove(file_path)

        break
    env.close()


if __name__ == "__main__":
    if len(sys.argv) == 9:
        max_steps = int(sys.argv[8])
    else:
        max_steps = 32

    if len(sys.argv) == 8:
        map_type = sys.argv[1]
        map_size_from = int(sys.argv[2])
        map_size_to = int(sys.argv[3])
        planner = sys.argv[4]
        use_fluents_map = sys.argv[5] == "True"
        fold = int(sys.argv[6])
        timeout = int(sys.argv[7])

        if (
            not os.path.isdir(f"{os.getcwd()}/dataset/{map_size_from}X{map_size_from}")
            or not os.path.isdir(f"{os.getcwd()}/dataset/{map_size_to}X{map_size_to}")
            or map_type not in ["advanced"]
            or map_size_from >= map_size_to
            or planner not in ["FF"]
            or fold not in list(range(5))
            or max_steps % 32 != 0
        ):
            print("Please provide valid command-line argument.")
            print(
                "Example: python playground_nsam_trans.py map_type[advanced] from_size[M1] to_size[M2] solver[FF] use_fluents_map[True/False] fold[0-4] time_out[seconds] optional_max_steps[32*X]"
            )
        else:
            main(
                map_type,
                map_size_from,
                map_size_to,
                planner,
                use_fluents_map,
                fold,
                timeout,
                max_steps,
            )
    else:
        print("Please provide a variable as a command-line argument.")
        print(
            "Example: python playground_nsam_trans.py map_type[advanced] from_size[M1] to_size[M2] solver[FF] use_fluents_map[True/False] fold[0-4] time_out[seconds] optional_max_steps[32*X]"
        )
