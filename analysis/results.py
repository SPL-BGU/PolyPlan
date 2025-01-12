import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import analysis
import seaborn as sns
from itertools import cycle


def process_task(task_folder):
    subdirs = ["10", "42", "63", "123", "1997"]
    files = ["Hybrid", "PPO"]

    columns_to_remove = [
        "first to goal",
        "who to min",
        "avg length w/o NSAM",
        "min length w/o NSAM",
    ]

    data = {file: [] for file in files}

    for subdir in subdirs:
        for file_name in files:
            if file_name == "Hybrid":
                df = analysis.analyze_hybrid(os.path.join(task_folder, subdir))
                if df is not None:
                    df = df.drop(columns=columns_to_remove, errors="ignore")
            elif file_name == "PPO":
                df = analysis.analyze_ppo(os.path.join(task_folder, subdir))
            if df is not None:
                data[file_name].append(df)
            else:
                # print("No data found for", os.path.join(task_folder, subdir, file_name))
                pass

    return data


def plot_results(data, task_name, output_dir, plot_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    line_styles = cycle(["-", "--", "-.", ":", "--", "-.", ":"])
    scatter_styles = cycle(["o", "^", "s"])
    colors = sns.color_palette("colorblind")
    color_cycle = cycle(colors)

    # Create a figure for the plots
    plt.figure(figsize=(10, 6))

    table_data = {}

    for file_name, dfs in data.items():
        if not dfs:
            continue

        # Concatenate and group the data
        grouped = (
            pd.concat(dfs, ignore_index=True)
            .groupby("map")
            .agg(
                {
                    "episodes to goal": ["mean", "std"],
                    "avg score": ["mean", "std"],
                    "max score": ["mean", "std"],
                    "avg length": ["mean", "std"],
                    "min length": ["mean", "std"],
                    "steps to goal": ["mean", "std"],
                }
            )
        )

        # Plot average length with shaded standard deviation
        mean = grouped[plot_type]["mean"]
        std = grouped[plot_type]["std"]
        x = grouped.index

        # plt.errorbar(
        #     x,
        #     mean,
        #     # yerr=std,  # Assuming 'std' holds the standard deviation values
        #     label=f"{file_name}",
        #     fmt=next(scatter_styles),  # 'o' specifies circle markers for scatter plot
        #     linestyle="None",  # No connecting lines between points
        #     color=next(color_cycle),
        #     linewidth=2,
        #     capsize=5,  # Add caps to the error bars
        # )

        # Plot mean scatter
        # window_size = 5
        # mean_smoothed = np.convolve(
        #     mean, np.ones(window_size) / window_size, mode="valid"
        # )
        # x_smoothed = x[: len(mean_smoothed)]
        # mean = mean_smoothed
        # x = x_smoothed

        # plt.scatter(
        #     x_smoothed,
        #     mean_smoothed,
        #     label=f"{file_name}",
        #     color=next(color_cycle),
        #     s=50,  # You can adjust the size of the scatter points with the 's' parameter
        #     marker=next(
        #         scatter_styles
        #     ),  # You can also specify different marker styles like 'o', '^', 's', etc.
        # )

        # Plot mean line
        plt.plot(
            x,
            mean,
            label=f"{file_name}",
            linestyle=next(line_styles),
            color=next(color_cycle),
            linewidth=2,
        )

        # Fill the area between the lines with standard deviation
        if "max" not in plot_type and "goal" not in plot_type:
            if "length" in plot_type:
                if output_dir.split("/")[1] == "PogoStick":
                    max_stepts = 1500
                else:
                    max_stepts = 200
            else:
                max_stepts = 1
            plt.fill_between(
                x,
                (mean - std).apply(lambda x: 0 if x < 0 else x),
                (mean + std).apply(lambda x: max_stepts if x > max_stepts else x),
                alpha=0.2,
                color=plt.gca().lines[-1].get_color(),
            )

        table_data[f"{file_name}"] = mean

    # Output the DataFrame to a CSV file
    table_data["Map"] = x
    df = pd.DataFrame(table_data)
    df.set_index("Map", inplace=True)
    output_file_path = f"{output_dir}/{task_name}.csv"
    df.to_csv(output_file_path)

    # Add labels and title
    plt.xlabel("Map", fontsize=14)
    plt.ylabel(plot_type.title(), fontsize=14)
    plt.title(f"{task_name} - Combined Results", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        os.path.join(output_dir, f"{task_name}.png"),
        bbox_inches="tight",
    )
    plt.close()


def main(results_dir, output_dir, plot_type):
    # task_folders = ["PogoStick", "WoodenSword"]
    task_folders = ["PogoStick"]
    # map_folders = ["6X6"]
    map_folders = ["6X6", "10X10", "15X15"]

    for task in task_folders:
        for map_folder in map_folders:
            task_folder = os.path.join(results_dir, task, map_folder)
            task_name = os.path.basename(task_folder)

            data = process_task(task_folder)
            plot_results(
                data, task_name, os.path.join(output_dir, task, plot_type), plot_type
            )


if __name__ == "__main__":
    results_dir = "results"
    output_dir = "output"

    for plot_type in [
        "avg length",
        "min length",
        "max score",
        "steps to goal",
        "episodes to goal",
    ]:
        main(results_dir, output_dir, plot_type)
