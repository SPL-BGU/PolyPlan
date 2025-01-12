import os


def check_task_folder(task_folder_path):
    expected_dirs = ["10", "42", "63", "123", "1997"]
    expected_files = ["Hybrid.csv", "PPO.csv"]

    count = 0
    for dir_name in expected_dirs:
        dir_path = os.path.join(task_folder_path, dir_name)
        if not os.path.isdir(dir_path):
            print(f"Missing directory: {dir_path}")
            continue

        for file_name in expected_files:
            file_path = os.path.join(dir_path, file_name)
            if not os.path.isfile(file_path):
                print(f"Missing file: {file_path}")
                count += 1

    if count == 0:
        print(f"All files and directories are present in {task_folder_path}")
    else:
        print(f"Missing {count} files/directories in {task_folder_path}")


def check_results_directory(results_dir):
    task_folders = ["PogoStick", "WoodenSword"]
    map_folders = ["6X6", "10X10", "15X15"]

    for task_folder in task_folders:
        for map_folder in map_folders:
            task_map_folder_path = os.path.join(results_dir, task_folder, map_folder)
            check_task_folder(task_map_folder_path)


def main():
    results_dir = "results"
    check_results_directory(results_dir)


if __name__ == "__main__":
    main()
