import os
import pandas as pd


def analyze_hybrid(folder_path: str):
    if not os.path.exists(os.path.join(folder_path, "Hybrid.csv")):
        return None

    # Step 1: Read CSV and Text Files
    df_hybrid = pd.read_csv(os.path.join(folder_path, "Hybrid.csv"))

    # Rename the first column to 'episodes'
    df_hybrid = df_hybrid.rename(columns={"Unnamed: 0": "episodes"})

    # Add 'log' column to df_hybrid with the log_values
    with open(os.path.join(folder_path, "did_plan.txt"), "r") as file:
        log_values = [line.strip() for line in file.readlines()]
    df_hybrid["log"] = log_values[: len(df_hybrid)]

    # Step 2: Add Cumulative Length Column
    df_hybrid["cumulative length"] = df_hybrid["length"].cumsum()

    # Step 3: Initialize an empty DataFrame for df_table
    df_table = pd.DataFrame(
        columns=[
            "map",
            "episodes to goal",
            "avg score",
            "max score",
            "avg length",
            "min length",
            "first to goal",
            "steps to goal",
            "who to min",
            "avg length without NSAM",
            "min length without NSAM",
        ]
    )

    # Step 4: Populate df_table with Metrics for Each Unique Iteration
    unique_iterations = df_hybrid["iteration"].unique()

    for iteration in unique_iterations:
        df_iter = df_hybrid[df_hybrid["iteration"] == iteration]

        map_id = int(iteration)
        episodes_to_goal = (
            df_iter.loc[df_iter["reward"] > 0, "episodes"].iloc[0] - df_iter.index[0]
            if (df_iter["reward"] > 0).any()
            else 4
        )
        goal_index = (
            (df_iter.loc[df_iter["reward"] > 0].index[0] - df_iter.index[0])
            if (df_iter["reward"] > 0).any()
            else -2
        )
        steps_to_goal = df_iter[: goal_index + 1]["length"].sum()
        avg_score = df_iter["reward"].mean()
        max_score = df_iter["reward"].max()
        avg_length = df_iter["length"].mean()
        min_length = df_iter["length"].min()
        first_to_goal = (
            "NSAM"
            if (df_iter["reward"] > 0).any()
            and df_iter.loc[df_iter["reward"] > 0, "log"].iloc[0]
            in ["plan found", "plan found by shorten"]
            else "PPO"
        )
        who_to_min = (
            "NSAM"
            if df_iter.loc[df_iter["length"] == min_length, "log"].iloc[0]
            in ["plan found", "plan found by shorten"]
            else "PPO"
        )
        wo_nsam = df_iter.loc[
            ~df_iter["log"].isin(["plan found", "plan found by shorten"]), "length"
        ]
        avg_length_wo_NSAM = wo_nsam.mean()
        min_length_wo_NSAM = wo_nsam.min()

        df_table.loc[len(df_table)] = {
            "map": map_id,
            "episodes to goal": episodes_to_goal,
            "avg score": avg_score,
            "max score": max_score,
            "avg length": avg_length,
            "min length": min_length,
            "first to goal": first_to_goal,
            "steps to goal": steps_to_goal,
            "who to min": who_to_min,
            "avg length without NSAM": avg_length_wo_NSAM,
            "min length without NSAM": min_length_wo_NSAM,
        }

    return df_table


def analyze_ppo(folder_path: str):
    if not os.path.exists(os.path.join(folder_path, "PPO.csv")):
        return None

    # Step 1: Read CSV and Text Files
    df_ppo = pd.read_csv(os.path.join(folder_path, "PPO.csv"))

    # Rename the first column to 'episodes'
    df_ppo = df_ppo.rename(columns={"Unnamed: 0": "episodes"})

    # Step 2: Add Cumulative Length Column
    df_ppo["cumulative length"] = df_ppo["length"].cumsum()

    # Step 3: Initialize an empty DataFrame for df_table
    df_table = pd.DataFrame(
        columns=[
            "map",
            "episodes to goal",
            "avg score",
            "max score",
            "avg length",
            "min length",
            "steps to goal",
        ]
    )

    # Step 4: Populate df_table with Metrics for Each Unique Iteration
    unique_iterations = df_ppo["iteration"].unique()

    for iteration in unique_iterations:
        df_iter = df_ppo[df_ppo["iteration"] == iteration]

        map_id = int(iteration)
        episodes_to_goal = (
            df_iter.loc[df_iter["reward"] > 0, "episodes"].iloc[0] - df_iter.index[0]
            if (df_iter["reward"] > 0).any()
            else 4
        )
        goal_index = (
            (df_iter.loc[df_iter["reward"] > 0].index[0] - df_iter.index[0])
            if (df_iter["reward"] > 0).any()
            else -2
        )
        steps_to_goal = df_iter[: goal_index + 1]["length"].sum()
        avg_score = df_iter["reward"].mean()
        max_score = df_iter["reward"].max()
        avg_length = df_iter["length"].mean()
        min_length = df_iter["length"].min()

        df_table.loc[len(df_table)] = {
            "map": map_id,
            "episodes to goal": episodes_to_goal,
            "avg score": avg_score,
            "max score": max_score,
            "avg length": avg_length,
            "min length": min_length,
            "steps to goal": steps_to_goal,
        }

    return df_table


if __name__ == "__main__":
    path = "results/PogoStick/6X6/63"
    print(analyze_hybrid(path))
    # print(analyze_ppo(path))
