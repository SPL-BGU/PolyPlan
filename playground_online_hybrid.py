import os
import sys

import time
from stable_baselines3.common.evaluation import evaluate_policy

from envs import MaskedMinecraft

from agents import ExploringSam, HybridPPO
from polycraft_policy import PolycraftMaskedPPOPolicy

from gym.wrappers import RecordEpisodeStatistics

from utils import Logger

import numpy as np
import random
import torch

SEED = 63
np.random.seed(SEED)  # random seed for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)


def main(map_type, map_size, use_fluents_map, steps_per_episode, steps_per_map):
    if map_type == "basic":
        env_index = 0  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft, 3: MaskedMinecraft
    elif map_type == "advanced":
        env_index = 2  # 0: BasicMinecraft, 1: IntermediateMinecraft, 2: AdvancedMinecraft, 3: MaskedMinecraft

    minecraft = MaskedMinecraft

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
            visually=False,
            start_pal=True,
            keep_alive=False,
            max_steps=steps_per_episode,
        )
    elif map_type == "advanced":
        env = minecraft(
            visually=False,
            start_pal=True,
            keep_alive=False,
            max_steps=steps_per_episode,
            map_size=map_size,
        )

    map_size = f"{map_size}X{map_size}"
    maps = 25

    train_idx = list(range(maps))

    output_directory_path = f"{os.getcwd()}/dataset/{map_size}"

    # make log directory
    postfix = f"PogoStick/HybridPPO/{map_type}_{map_size}"
    logdir, models_dir = Logger.create_logdir(postfix, indexing=True)

    rec_dir = f"{logdir}/solutions"
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    renv = RecordEpisodeStatistics(env, deque_size=10000)

    exploring_sam = ExploringSam(
        env,
        domain=p_domain,
        problem=p_problem,
        fluents_map=fluents_map,
        save_interval=1,
        output_dir=rec_dir,
    )
    model = HybridPPO(
        exploring_sam,
        use_fluents_map=use_fluents_map,
        shortest_possible_plan=5,
        policy=PolycraftMaskedPPOPolicy,
        env=renv,
        verbose=1,
        n_steps=steps_per_episode,
        batch_size=steps_per_episode,
        stats_window_size=1,
        ent_coef=0.01,
        gamma=0.999,
        vf_coef=0.65,
        max_grad_norm=1.0,
        tensorboard_log=logdir,
        seed=SEED,
    )

    callback = Logger.RecordTrajectories(output_dir=rec_dir)

    for index, problem_index in enumerate(train_idx):
        t1 = time.time()
        domain_path = f"{output_directory_path}/map_instance_{problem_index}.json"
        env.set_domain(domain_path)

        renv.reset()

        problem = (
            f"{output_directory_path}/{map_type}_map_instance_{problem_index}.pddl"
        )
        model.update_explorer_problem(problem)

        model.learn(
            total_timesteps=steps_per_map,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        model.exploring_sam = None
        model.save(f"{models_dir}/{index}.zip")
        model.exploring_sam = exploring_sam
        t2 = time.time()
        print(f"Time to train: {t2 - t1}")

    env.close()


if __name__ == "__main__":
    if len(sys.argv) == 6:
        map_type = sys.argv[1]
        map_size = int(sys.argv[2])
        use_fluents_map = sys.argv[3] == "True"
        steps_per_episode = int(sys.argv[4])
        steps_per_map = int(sys.argv[5])

        if (
            not os.path.isdir(f"{os.getcwd()}/dataset/{map_size}X{map_size}")
            or map_type not in ["basic", "advanced"]
            or steps_per_map % steps_per_episode != 0
        ):
            print("Please provide valid command-line argument.")
            print(
                "Example: python playground_online_hybrid.py map_type[basic/advanced] map_size[<int>] use_fluents_map[True/False] steps_per_episode[<int>] steps_per_map[<int>]"
            )
        else:
            main(map_type, map_size, use_fluents_map, steps_per_episode, steps_per_map)
    else:
        print("Please provide a variable as a command-line argument.")
        print(
            "Example: python playground_online_hybrid.py map_type[basic/advanced] map_size[<int>] use_fluents_map[True/False] steps_per_episode[<int>] steps_per_map[<int>]"
        )
