import os
import pandas as pd


def determine_whosolve(row):
    if "PPO" in row["Alg."]:
        return "PPO"
    elif "1" in row["Alg."]:
        if "but" in row["Planner"]:
            return "PPO"
        elif ("plan found" in row["Planner"]) or ("by shorten" in row["Planner"]):
            return "Shorter"
        else:
            return "PPO"
    elif "3" in row["Alg."]:
        if "plan found" in row["Planner"]:
            if "but" in row["Planner"]:
                return "PPO"
            elif "by shorten" in row["Planner"]:
                return "Shorter"
            else:
                return "Planner"
        else:
            return "PPO"
    else:
        return "PPO"


def process_task(task_folder, task, map_folder):
    subdirs = ["10", "42", "63", "123", "1997"]

    directory_path = os.path.join(task_folder, os.listdir(task_folder)[0])
    files = [
        name
        for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name))
    ]

    data = []

    for subdir in subdirs:
        for file_name in files:

            # filter out the files we want to plot
            # if file_name in ["Hybrid"]:
            #     continue

            current_path = os.path.join(task_folder, subdir, file_name)
            if "Hybrid" in file_name:
                csv_path = os.path.join(current_path, "Hybrid.csv")
            elif "PPO" in file_name:
                csv_path = os.path.join(current_path, "PPO.csv")

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

                if "Hybrid" in file_name:
                    with open(os.path.join(current_path, "did_plan.txt"), "r") as file:
                        planner = [line.strip() for line in file.readlines()]

                    df["Planner"] = planner

                    with open(
                        os.path.join(current_path, "time_to_plan.txt"), "r"
                    ) as file:
                        planner = [line.strip() for line in file.readlines()]
                    with open(
                        os.path.join(current_path, "total_nsam_time.txt"), "r"
                    ) as file:
                        learner = [line.strip() for line in file.readlines()]

                    # fix offset if needed
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
                            if total_length > max_length:
                                planner.pop(i - removed)
                                learner.pop(i - removed)
                                removed += 1
                            total_length = 0
                        total_length += length
                    if total_length > max_length:
                        planner.pop(-1)
                        learner.pop(-1)
                    for i in range(len(planner)):
                        if planner[i] == "-1":
                            planner[i] = "N/A"
                            learner[i] = "N/A"

                    df["Runtime planner"] = planner
                    df["Runtime learner"] = learner
                elif "PPO" in file_name:
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
                df["WhoSolve"] = df.apply(determine_whosolve, axis=1)

                # Append the data
                data.append(df)
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()


def main(results_dir):
    task_folders = ["WoodenSword", "PogoStick"]
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
                    "Fold": "Seed",
                },
                inplace=True,
            )
            table = pd.concat([table, data], ignore_index=True)

    table.to_csv("total.csv")


if __name__ == "__main__":
    results_dir = "results"
    main(results_dir)
