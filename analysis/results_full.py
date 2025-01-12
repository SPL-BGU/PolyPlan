import os
import pandas as pd


def process_task(task_folder, task, map_folder):
    subdirs = ["10", "42", "63", "123", "1997"]
    files = ["Hybrid", "PPO"]

    data = []

    for subdir in subdirs:
        for file_name in files:
            if file_name == "Hybrid":
                csv_path = os.path.join(task_folder, subdir, "Hybrid.csv")
            elif file_name == "PPO":
                csv_path = os.path.join(task_folder, subdir, "PPO.csv")

            if not os.path.exists(csv_path):
                df = None
            else:
                df = pd.read_csv(csv_path).rename(columns={"Unnamed: 0": "episodes"})

            if df is not None:
                # Add necessary metadata columns
                df["Domain"] = task
                df["MapSize"] = map_folder
                df["Fold"] = subdir
                df["Alg."] = file_name

                if file_name == "Hybrid":
                    with open(
                        os.path.join(task_folder, subdir, "did_plan.txt"), "r"
                    ) as file:
                        planner = [line.strip() for line in file.readlines()]
                    df["Planner"] = planner

                    with open(
                        os.path.join(task_folder, subdir, "time_to_plan.txt"), "r"
                    ) as file:
                        planner = [line.strip() for line in file.readlines()]
                    with open(
                        os.path.join(task_folder, subdir, "total_nsam_time.txt"), "r"
                    ) as file:
                        learner = [line.strip() for line in file.readlines()]
                    current_map = 0
                    total_length = 0
                    max_length = 6000 if task == "PogoStick" else 800
                    removed = 0
                    for i, (length, map) in enumerate(
                        zip(df["length"], df["iteration"])
                    ):
                        map = int(map)
                        if current_map != map:
                            current_map = map
                            if total_length != max_length:
                                planner.pop(i - removed)
                                learner.pop(i - removed)
                                removed += 1
                            total_length = 0
                        total_length += length
                    if total_length != max_length:
                        planner.pop(-1)
                        learner.pop(-1)
                    for i in range(len(planner)):
                        if planner[i] == "-1":
                            planner[i] = "N/A"
                            learner[i] = "N/A"
                    df["Runtime planner"] = planner
                    df["Runtime learner"] = learner
                elif file_name == "PPO":
                    df["Planner"] = "N/A"
                    df["Runtime planner"] = "N/A"
                    df["Runtime learner"] = "N/A"

                # Calculate and add the "Current Min Length" column
                min_lengths = []
                current_map = -1
                for length, map in zip(df["length"], df["iteration"]):
                    if map != current_map:
                        current_map = map
                        current_min_length = float("inf")
                    current_min_length = min(
                        current_min_length, length
                    )  # Update running minimum
                    min_lengths.append(current_min_length)
                df["Current Min Length"] = min_lengths  # Add the running min column

                # Append the data
                data.append(df)
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()


def main(results_dir):
    # task_folders = ["WoodenSword", "PogoStick"]
    task_folders = ["PogoStick"]
    map_folders = ["6X6", "10X10", "15X15"]
    table = pd.DataFrame()

    for task in task_folders:
        for map_folder in map_folders:
            task_folder = os.path.join(results_dir, task, map_folder)

            data = process_task(task_folder, task, map_folder)
            data.rename(
                columns={
                    "iteration": "map",
                    "reward": "Solved or not",
                    "episodes": "episode",
                },
                inplace=True,
            )
            table = pd.concat([table, data], ignore_index=True)

    table.to_csv("total.csv")


if __name__ == "__main__":
    results_dir = "results"
    main(results_dir)
