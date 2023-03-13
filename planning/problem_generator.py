from pathlib import Path
from typing import List

from common import get_problem_template

TEMPLATE_FILE_PATH = Path("problem_template.pddl")


def generate_instance(
    instance_name: str,
    agent_position: int,
    tree_positions: List[int],
    table_positions: int,
) -> str:
    """Generate a single planning problem instance.

    :param instance_name: the name of the problem instance.
        :param agent_position: the number of counters in the problem.
        :param tree_positions: the maximal integer value.
    :return: the string representing the planning problem.
    """
    template = get_problem_template(TEMPLATE_FILE_PATH)
    template_mapping = {
        "instance_name": instance_name,
        "cell_list": " ".join(
            [f"cell{i}" for i in range(30 * 30) if i != table_positions]
        ),
        "agent_position_initial_values": f"(position cell{agent_position})",
        "air_cell_initial_values": "\n".join(
            [
                f"(= (cell_type cell{i}) 0)"
                for i in range(30 * 30)
                if i not in tree_positions and i != table_positions
            ]
        ),
        "tree_cell_initial_values": "\n".join(
            [f"(= (cell_type cell{i}) 1)" for i in tree_positions]
        ),
    }
    return template.substitute(template_mapping)


def main():
    instance_name = "advanced_minecraft_problem.pddl"
    agent_position = 26 * 30 + 26  # (27,27)
    tree_positions = [
        16 * 30 + 2,
        1 * 30 + 7,
        9 * 30 + 9,
        25 * 30 + 17,
        14 * 30 + 22,
    ]  # (3,17) (8,2) (10,10) (18,26) (23,15)
    table_positions = 19 * 30 + 19  # (20,20)

    with open(instance_name, "wt") as problem_file:
        problem_file.write(
            generate_instance(
                f"instance_{agent_position}_{table_positions}",
                agent_position,
                tree_positions,
                table_positions,
            )
        )


if __name__ == "__main__":
    main()
