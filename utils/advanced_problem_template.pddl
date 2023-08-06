(define (problem ${instance_name})
	(:domain PolyCraft)
	(:objects
		${cell_list} - cell
	)
	(:init ${agent_position} ${air_cells} ${tree_cells}
		(= (cell_type crafting_table) 2) ${count_log_in_inventory_initial} ${count_planks_in_inventory_initial} ${count_stick_in_inventory_initial} ${count_sack_polyisoprene_pellets_in_inventory_initial} ${count_tree_tap_in_inventory_initial}
		(= (count_wooden_pogo_stick_in_inventory) 0)
	)
	(:goal
		(and
			(>= (count_wooden_pogo_stick_in_inventory) 1)
		)
	)
)