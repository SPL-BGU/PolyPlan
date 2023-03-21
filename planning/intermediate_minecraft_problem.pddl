; PolyCraft intermediate problem

(define (problem intermediate)
	(:domain PolyCraft)
	(:objects
		cell1 cell2 cell3 cell4 cell5
	)
	(:init
		(position cell0)

		; Map
		(= (cell_type cell1) 1)
		(= (cell_type cell2) 1)
		(= (cell_type cell3) 1)
		(= (cell_type cell4) 1)
		(= (cell_type cell5) 1)

		(= (cell_type cell0) 2)

		; Items
		(= (count_log_in_inventory) 0)
		(= (count_planks_in_inventory) 0)
		(= (count_stick_in_inventory) 0)
		(= (count_sack_polyisoprene_pellets_in_inventory) 0)
		(= (count_tree_tap_in_inventory) 0)
		(= (count_wooden_pogo_stick_in_inventory) 0)
	)
	(:goal
		(and
			(= (count_wooden_pogo_stick_in_inventory) 1)
		)
	)
)