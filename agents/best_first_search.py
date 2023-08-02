from agents.polycraft_agent import PolycraftAgent
import networkx as nx
from random import choice
import random


class BFS(PolycraftAgent):
    """Online Best-First Search agent with mask for invalid actions"""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = self.env.action_space.n
        self.graph = nx.DiGraph()

        self.frontier = set()
        self.current_successors = []
        self.visited = set()

        # init graph
        self.start_state = str(env.state)

        # for update useless actions
        self.last_state = (None, 0)
        self.last_action = None

        # for update useless states
        self.pre_state = None
        self.pre_action = None

    def priorite_calculation(self, state_action: tuple) -> int:
        """calculate the priority of state and action, higher is better
        state_action: tuple of (state, action)"""
        return 0

    def top_priority(self, all_set: set) -> list:
        """return a list of the top priority values from a set"""
        top_list = []

        max_priority = 0
        for state_action in all_set:
            priority = self.priorite_calculation(state_action)
            if priority < max_priority:
                continue
            elif priority > max_priority:
                max_priority = priority
            top_list.append((priority, state_action))

        top_list = [tup for priority, tup in top_list if priority == max_priority]
        return top_list

    def choosing(self, top_list: list) -> tuple:
        """return a tuple of (state, action)"""
        return choice(top_list)

    def choose_action(self, state=None) -> int:
        """choose random action from the top priority actions"""
        state = str(state)
        self.check_state_exist(state)

        if state == self.start_state:  # explore from frontier
            frontier = self.top_priority(set(self.graph.neighbors(state)))
            state_action = self.choosing(frontier)
            path_to_state = self.path_to_state(state_action[0])
            self.walk_in_path(path_to_state)
        else:  # explore from near neighbors
            if len(self.current_successors) == 0:
                self.last_state = (None, 0)
                return 0
            state_action = self.choosing(self.current_successors)

        self.frontier.discard(state_action)
        self.last_state = state_action
        return state_action[1]

    def check_state_exist(self, state):
        if state not in self.visited:
            self.visited.add(state)
            self.graph.add_edge(self.last_state, state)
            for i in range(self.action_space):
                next_state = (state, i)
                self.graph.add_edge(state, next_state)
                self.frontier.add(next_state)
        elif self.last_state[0] == state:  # we got back to the same state
            # invalid action masking
            self.graph.remove_node(self.last_state)

        self.current_successors = self.top_priority(set(self.graph.neighbors(state)))

    def path_to_state(self, state):
        try:
            return nx.shortest_path(self.graph, str(self.env.state), str(state))
        except nx.NetworkXNoPath:
            return None

    def walk_in_path(self, path):
        for tup in path:
            if type(tup) == tuple:
                self.env.step(tup[1])
