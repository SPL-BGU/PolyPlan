import os
import json
import random
import shutil
from typing import List
from pathlib import Path

from pddl_plus_parser.problem_generators import get_problem_template


def basic_pddl_minecraft_generate(
    instance_name: str,
    trees_in_map: int,
    count_log_in_inventory: int,
    count_planks_in_inventory: int,
    count_stick_in_inventory: int,
    count_sack_polyisoprene_pellets_in_inventory: int,
    count_tree_tap_in_inventory: int,
) -> str:
    """
    Generate a single basic planning problem instance.

    :instance_name: the name of the problem instance.
    :trees_in_map: the number of trees in the map.
    :count_log_in_inventory: the number of logs in the inventory.
    :count_planks_in_inventory: the number of planks in the inventory.
    :count_stick_in_inventory: the number of sticks in the inventory.
    :count_sack_polyisoprene_pellets_in_inventory: the number of sacks of polyisoprene pellets in the inventory.
    :count_tree_tap_in_inventory: the number of tree taps in the inventory.
    """
    template = get_problem_template(Path("utils/basic_minecraft_template.pddl"))
    template_mapping = {
        "instance_name": instance_name,
        "trees_in_map_initial": f"(= (trees_in_map) {trees_in_map})",
        "count_log_in_inventory_initial": f"(= (count_log_in_inventory) {count_log_in_inventory})",
        "count_planks_in_inventory_initial": f"(= (count_planks_in_inventory) {count_planks_in_inventory})",
        "count_stick_in_inventory_initial": f"(= (count_stick_in_inventory) {count_stick_in_inventory})",
        "count_sack_polyisoprene_pellets_in_inventory_initial": f"(= (count_sack_polyisoprene_pellets_in_inventory) {count_sack_polyisoprene_pellets_in_inventory})",
        "count_tree_tap_in_inventory_initial": f"(= (count_tree_tap_in_inventory) {count_tree_tap_in_inventory})",
    }
    return template.substitute(template_mapping)


def advanced_pddl_minecraft_generate(
    instance_name: str,
    map_size: int,
    crafting_table_cell: int,
    agent_position: int,
    tree_positions: List[int],
    count_log_in_inventory: int,
    count_planks_in_inventory: int,
    count_stick_in_inventory: int,
    count_sack_polyisoprene_pellets_in_inventory: int,
    count_tree_tap_in_inventory: int,
) -> str:
    """
    Generate a single advanced planning problem instance.

    :instance_name: the name of the problem instance.
    :map_size: the size of the map.
    :crafting_table_cell: the cell of the crafting table.
    :agent_position: the cell of the agent.
    :tree_positions: the cells of the trees.
    :count_log_in_inventory: the number of logs in the inventory.
    :count_planks_in_inventory: the number of planks in the inventory.
    :count_stick_in_inventory: the number of sticks in the inventory.
    :count_sack_polyisoprene_pellets_in_inventory: the number of sacks of polyisoprene pellets in the inventory.
    :count_tree_tap_in_inventory: the number of tree taps in the inventory.
    """
    template = get_problem_template(Path("utils/advanced_problem_template.pddl"))
    template_mapping = {
        "instance_name": instance_name,
        "cell_list": " ".join(
            [f"cell{i}" for i in range(map_size) if i != crafting_table_cell]
        ),
        "agent_position": f"(position cell{agent_position})",
        "air_cells": " ".join(
            [
                f"(= (cell_type cell{i}) 0)"
                for i in range(map_size)
                if i not in tree_positions and i != crafting_table_cell
            ]
        ),
        "tree_cells": " ".join([f"(= (cell_type cell{i}) 1)" for i in tree_positions]),
        "count_log_in_inventory_initial": f"(= (count_log_in_inventory) {count_log_in_inventory})",
        "count_planks_in_inventory_initial": f"(= (count_planks_in_inventory) {count_planks_in_inventory})",
        "count_stick_in_inventory_initial": f"(= (count_stick_in_inventory) {count_stick_in_inventory})",
        "count_sack_polyisoprene_pellets_in_inventory_initial": f"(= (count_sack_polyisoprene_pellets_in_inventory) {count_sack_polyisoprene_pellets_in_inventory})",
        "count_tree_tap_in_inventory_initial": f"(= (count_tree_tap_in_inventory) {count_tree_tap_in_inventory})",
    }
    return template.substitute(template_mapping)


def generate_problems(
    num_maps_to_generate: int,
    map_size: int,
    example_map_path: Path,
    output_directory_path: Path,
    compiled_json_file: Path,
) -> None:
    """
    Generate maps using the example map.

    :param num_maps_to_generate: number of maps to generate
    :param map_size: actual map size is map_size**2
    :param example_map_path: path to example map
    :param output_directory_path: path to output directory
    :param compiled_json_file: path to compiled json file
    """

    generator = lambda: [
        random.randint(2, map_size - 1),
        4,
        random.randint(2, map_size - 1),
    ]

    with open(example_map_path, "r") as map_file:
        map_json = json.load(map_file)
        for i in range(num_maps_to_generate):
            output_map_file_path = output_directory_path / f"map_instance_{i}.json"
            num_trees = random.randint(4, 5)
            tree_locations_in_map = []

            # crafting table location
            crafting_table_location_in_map = generator()
            objects_in_map = [
                {
                    "blockPos": crafting_table_location_in_map,
                    "blockName": "minecraft:crafting_table",
                }
            ]

            # agent location
            while (new_point := generator()) == crafting_table_location_in_map:
                pass
            agent_starting_location = new_point

            # trees in map
            for _ in range(num_trees):
                while (
                    new_point := generator()
                ) in tree_locations_in_map or new_point in [
                    crafting_table_location_in_map,
                    agent_starting_location,
                ]:
                    pass

                tree_locations_in_map.append(new_point)
                objects_in_map.append(
                    {
                        "blockPos": new_point,
                        "blockName": "tree",
                    }
                )

            # initial inventory
            initial_inventory = {
                "pos": [3, 4, 6],
                "name": "Add Items 1",
                "color": -256,
                "type": "ADD_ITEMS",
                "canProceed": False,
                "isDone": False,
                "completionTime": 0,
                "uuid": "b6ae0b79-e683-4c8a-b33c-39659562c617",
                "itemList": [],
            }

            items = [
                "minecraft:log",
                "minecraft:planks",
                "minecraft:stick",
                "polycraft:sack_polyisoprene_pellets",
                "polycraft:tree_tap",
            ]
            items_range = [64, 64, 64, 1, 1]
            items_count = {}

            for j, (item, item_range) in enumerate(zip(items, items_range)):
                num_items = random.randint(0, item_range)
                items_count[item] = num_items
                if num_items == 0:
                    continue
                initial_inventory["itemList"].append(
                    {
                        "slot": j,
                        "itemDef": {
                            "itemName": item,
                            "itemMeta": 0,
                            "count": num_items,
                        },
                    }
                )

            # update json
            map_json["features"][0]["pos"] = agent_starting_location
            map_json["features"][1]["pos2"] = [map_size + 1, 6, map_size + 1]
            map_json["features"][2]["blockList"] = objects_in_map
            map_json["features"][3]["pos2"] = [map_size + 1, 4, map_size + 1]
            map_json["features"].insert(5, initial_inventory)
            with open(output_map_file_path, "wt") as output:
                json.dump(map_json, output)

            shutil.copy(
                compiled_json_file,
                output_directory_path / f"map_instance_{i}.json2",
            )

            # generate basic minecraft pddl
            with open(
                output_directory_path / f"basic_map_instance_{i}.pddl", "wt"
            ) as problem_file:
                problem_file.write(
                    basic_pddl_minecraft_generate(
                        instance_name=f"instance_{i}",
                        trees_in_map=len(objects_in_map) - 1,
                        count_log_in_inventory=items_count["minecraft:log"],
                        count_planks_in_inventory=items_count["minecraft:planks"],
                        count_stick_in_inventory=items_count["minecraft:stick"],
                        count_sack_polyisoprene_pellets_in_inventory=items_count[
                            "polycraft:sack_polyisoprene_pellets"
                        ],
                        count_tree_tap_in_inventory=items_count["polycraft:tree_tap"],
                    )
                )

            # generate advanced minecraft pddl
            transform_to_cell = lambda x: (x[0] - 1) + ((x[2] - 1) * map_size)
            tree_positions = []
            for tree_location in tree_locations_in_map:
                tree_positions.append(transform_to_cell(tree_location))
            crafting_table_cell = transform_to_cell(crafting_table_location_in_map)
            agent_position = transform_to_cell(agent_starting_location)

            with open(
                output_directory_path / f"advanced_map_instance_{i}.pddl", "wt"
            ) as problem_file:
                problem_file.write(
                    advanced_pddl_minecraft_generate(
                        instance_name=f"instance_{i}",
                        map_size=map_size**2,
                        crafting_table_cell=crafting_table_cell,
                        agent_position=agent_position,
                        tree_positions=tree_positions,
                        count_log_in_inventory=items_count["minecraft:log"],
                        count_planks_in_inventory=items_count["minecraft:planks"],
                        count_stick_in_inventory=items_count["minecraft:stick"],
                        count_sack_polyisoprene_pellets_in_inventory=items_count[
                            "polycraft:sack_polyisoprene_pellets"
                        ],
                        count_tree_tap_in_inventory=items_count["polycraft:tree_tap"],
                    )
                )


if __name__ == "__main__":
    map_size = 5
    num_maps_to_generate = 100
    path = f"/home/benjamin/Projects/PolyPlan/dataset/{map_size}X{map_size}"
    if not os.path.exists(path):
        os.makedirs(path)

    generate_problems(
        num_maps_to_generate=num_maps_to_generate,
        map_size=map_size,
        example_map_path=Path(
            "/home/benjamin/Projects/pal/available_tests/pogo_nonov.json"
        ),
        output_directory_path=Path(path),
        compiled_json_file=Path(
            "/home/benjamin/Projects/pal/available_tests/pogo_nonov.json2"
        ),
    )
