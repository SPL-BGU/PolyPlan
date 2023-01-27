; PolyCraft basic problem

(define (problem basic)

(:domain PolyCraft)

(:init
    ; Map
    (= (trees_in_map) 4)

    ; Items
    (= (count_platinum_in_inventory) 0)
    (= (count_titanium_in_inventory) 0)
    (= (count_crafting_table_in_inventory) 0)
    (= (count_diamond_in_inventory) 0)
    (= (count_diamond_block_in_inventory) 0)
    (= (count_iron_pickaxe_in_inventory) 0)
    (= (count_key_in_inventory) 0)
    (= (count_log_in_inventory) 0)
    (= (count_planks_in_inventory) 0)
    (= (count_sapling_in_inventory) 0)
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