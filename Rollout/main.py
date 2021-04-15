import time
import copy
import gym
import gym_sokoban
import numpy as np
import pickle
import os

# Credit for this: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
from warnings import warn
import heapq
import random
import time


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(maze, start, end, allow_diagonal_movement=False):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    max_iterations = (10*len(maze[0]) * len(maze) // 2)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),
                            (-1, -1), (-1, 1), (1, -1), (1, 1),)

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
            # if we hit this point return the path such as it is
            # it will not contain the destination
            warn("giving up on pathfinding too many iterations")
            return return_path(current_node)

        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []

        for new_position in adjacent_squares:  # Adjacent squares

            # Get node position
            node_position = (
                current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = abs(child.position[0] - end_node.position[0]) + \
                abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None


def pre_processing(state_mat, target):
    # Represent all obstacles with 1, free space with 0
    for i in range(state_mat.shape[0]):
        for j in range(state_mat.shape[1]):
            # If free space or agent
            if state_mat[i][j] == 1 or state_mat[i][j] == 2 or state_mat[i][j] >= 5:
                state_mat[i][j] = 0
            elif state_mat[i][j] == 0 or state_mat[i][j] == 4:  # wall or box
                state_mat[i][j] = 1
    # print(target)
    state_mat[target] = 0
    return state_mat


delta_to_action = {
    (-1, 0): 1,
    (1, 0): 2,
    (0, -1): 3,
    (0, 1): 4}


def base_policy(state, return_list, agent_number):
    state_mat = state[0]
    state_targets = state[1]
    agent_id = 5+2*agent_number

    # Linear search
    indicies = np.argwhere((state_mat == agent_id) | (state_mat == agent_id+1))
    pos = (indicies[0][0], indicies[0][1])

    processed_state_mat = pre_processing(
        state_mat, state_targets[agent_number])
    # print("Agent ", agent_number, " pos ", pos)
    # print("Agent ", agent_number, "target", state_targets[agent_number])
    if state_targets[agent_number] == None:
        return [0] if return_list else 0
    path = astar(processed_state_mat, pos, state_targets[agent_number])

    actions = []
    for i in range(1, len(path)):
        actions.append(
            delta_to_action[(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])])
    # print(agent_number, actions)
    if len(actions) == 0:
        return [0] if return_list else 0
    return actions if return_list else actions[0]


def action_picker(env, prev_actions, state, num_of_agents, depth, num_of_steps):
    action_space = 5
    R = [0]*action_space
    for action in range(action_space):  # For every action
        # print("ACTION", action, "---------------------------------------------")
        new_state = copy.deepcopy(state)
        cached_state = (new_state[0], new_state[1], num_of_agents)
        env.reset(render_mode="raw", cached_state=cached_state,
                  num_of_agents=num_of_agents)
        next_actions = []
        for i in range(len(prev_actions)+1, num_of_agents):  # Iterates next robots
            next_actions.append(base_policy(copy.deepcopy(state), False, i))

        action_list = prev_actions + [action] + next_actions
        # Simulate
        done = False
        n = 0
        # print("START SIMULATION")
        targets = [(-1, -1)]*8
        path = [0]*8
        while n < depth:
            curr_state, reward, done, info = env.step(action_list, "raw")
            R[action] += reward
            if done:
                break
            action_list = [0]*num_of_agents
            for i in range(num_of_agents):
                if targets[i] != curr_state[1][i] or len(path[i]) == 0:
                    path[i] = base_policy(
                        copy.deepcopy(curr_state), True, i)
                    targets[i] = curr_state[1][i]
                action_list[i] = path[i].pop(0)
            n += 1

    return R


num_of_agents = 8
env = gym.make("Sokoban-v0")

actions_to_delta = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1)
}
agent_color = {0: "Red", 1: "Purple", 2: "Green", 3: "Deep blue",
               4: "Yellow", 5: "Light blue", 6: "Pink", 7: "Deep purple"}
number_of_tests = 0
number_of_successes = 0

state_vec = []
action_vec = []
while True:
    state = env.reset(render_mode="raw", num_of_agents=num_of_agents)
    # env.render()
    reward_tot = 0
    done = False
    num_of_steps = 0

    mini_state_vec = []
    mini_action_vec = []
    while not done:

        # input()
        agent_list = [i for i in range(num_of_agents)]
        number_of_shuffles = 0
        while True:
            action_list = []
            for i in agent_list:
                R = action_picker(
                    env, action_list, state, num_of_agents, 200, num_of_steps)
                #print(agent_color[i], "agents", "rewards", R)
                max_value = max(R)
                possible_actions = [
                    i for i, x in enumerate(R) if x == max_value]

                state_mat = state[0]
                state_targets = state[1]
                agent_id = 5+2*i

                # Linear search
                indicies = np.argwhere((state_mat == agent_id) |
                                       (state_mat == agent_id+1))
                pos = (indicies[0][0], indicies[0][1])
                if state_targets[i] == None:
                    action_list.append(random.choice(possible_actions))
                else:
                    path_lengths = []
                    for action in possible_actions:
                        delta = actions_to_delta[action]
                        new_mat = copy.deepcopy(state_mat)
                        processed_state_mat = pre_processing(
                            new_mat, state_targets[i])
                        new_pos = (pos[0]+delta[0], pos[1]+delta[1])
                        if processed_state_mat[new_pos] == 0:
                            path = astar(processed_state_mat,
                                         new_pos, state_targets[i])
                            path_lengths.append(len(path))
                        else:
                            path_lengths.append(100)

                    shortest_path_length_index = path_lengths.index(
                        min(path_lengths))
                    action_list.append(
                        possible_actions[shortest_path_length_index])
            #print("DECISION: ", action_list)
            cached_state = (state[0], state[1], num_of_steps)
            cached_state_copy = copy.deepcopy(cached_state)
            state = env.reset(render_mode="raw",
                              num_of_agents=num_of_agents, cached_state=cached_state_copy)
            state, reward, done, info = env.step(action_list, "raw")

            if reward < -8 and number_of_shuffles < 80:
                state = env.reset(render_mode="raw",
                                  num_of_agents=num_of_agents, cached_state=cached_state)
                random.shuffle(agent_list)
                number_of_shuffles += 1
                print(
                    "COLLISION DETECTED, SHUFFLING------------------------------------------------------------------")
                continue

            reward_tot += reward
            mini_state_vec.append(state)
            mini_action_vec.append(action_list)
            num_of_steps += 1
            break
        # env.render()
    print("Total reward: ", reward_tot)
    print("Number of steps: ", num_of_steps)
    interval = 5
    if reward_tot >= 11000-200*num_of_agents:
        state_vec = state_vec+mini_state_vec
        action_vec = action_vec+mini_action_vec
        number_of_successes += 1
        if number_of_successes % interval == 0:
            file_list = os.listdir("./Dataset")
            file_name = ""
            temp = ()
            if len(file_list) > 0:
                sorted(file_list)
                index = int(file_list[-1][4:])
                infile = open("./Dataset/data"+str(index), 'rb')
                prev = pickle.load(infile)
                infile.close()
                filename = "data"+str(index+1)
                temp = (prev[0]+state_vec, prev[1]+action_vec)
            else:
                filename = "data0"
                temp = (state_vec, action_vec)
            outfile = open("./Dataset/"+filename, 'wb')
            pickle.dump(temp, outfile)
            outfile.close()

    number_of_tests += 1
    print(number_of_tests, " ", 100*number_of_successes /
          number_of_tests, "%", "success rate")
