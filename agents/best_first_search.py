from agents import PolycraftAgent
import networkx as nx
from random import choice


class BFS(PolycraftAgent):
    """
    Online Best-First Search agent with mask for invalid actions

    :all_successors: set of all possible successors
    :frontier: list of the current successors
    :node: tuple of (state, action)
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = self.env.action_space.n
        self.graph = nx.DiGraph()
        self.start_state = str(env.state)

        self.all_successors = set()
        self.frontier = []
        self.visited = set()
        self.last_node = "Start"

    def priorite_calculation(self, node: tuple) -> int:
        """
        calculate the priority of state and action, higher is better
        :node: tuple of (state, action)
        """
        return 0

    def top_priority(self, node_set: set) -> list:
        """
        return a list of the top priority values from a set
        :node_set: set of nodes (tuples of (state, action))
        """
        top_list = []

        max_priority = 0
        for node in node_set:
            priority = self.priorite_calculation(node)
            if priority < max_priority:
                continue
            elif priority > max_priority:
                max_priority = priority
            top_list.append((priority, node))

        top_list = [node for priority, node in top_list if priority == max_priority]
        return top_list

    def choosing(self, top_list: list) -> tuple:
        """select a node (state, action) from the top priority list"""
        return choice(top_list)

    def choose_action(self, state=None) -> int:
        """choose the next action to perform"""
        state = str(state)
        self.check_state_exist(state)

        if state == self.start_state:  # explore from all_successors
            frontier = self.top_priority(self.all_successors)
            node = self.choosing(frontier)
            self.get_to_state(node[0])
        else:  # explore from frontier
            if len(self.frontier) == 0:
                self.last_node = (None, 0)
                return 0
            node = self.choosing(self.frontier)

        self.all_successors.discard(node)
        self.last_node = node
        return node[1]

    def check_state_exist(self, state):
        """
        add state to the graph if it doesn't exist
        if it does exist make a mask for invalid actions
        """
        if state not in self.visited:
            self.visited.add(state)
            self.graph.add_edge(self.last_node, state)
            for i in range(self.action_space):
                node = (state, i)
                self.graph.add_edge(state, node)
                self.all_successors.add(node)
        elif self.last_node[0] == state:  # we got back to the same state
            # invalid action masking
            self.graph.remove_node(self.last_node)

        self.frontier = self.top_priority(set(self.graph.neighbors(state)))

    def path_to_state(self, state):
        try:
            return nx.shortest_path(self.graph, str(self.env.state), str(state))
        except nx.NetworkXNoPath:
            return None

    def walk_in_path(self, path):
        for node in path:
            if type(node) == tuple:
                self.env.step(node[1])

    def get_to_state(self, state):
        path = self.path_to_state(state)
        self.walk_in_path(path)
