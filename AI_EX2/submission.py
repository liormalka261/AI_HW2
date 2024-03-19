from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    def h(env: WarehouseEnv, robot_id: int):
        robot = env.get_robot(robot_id)
        opponent = env.get_robot((robot_id + 1) % 2)
        #steps_left = env.num_steps // 2
        #last_charge = False # if we wont go back to charge
        #closest_station = env.charge_stations[0] if manhattan_distance(robot, env.charge_stations[0]) <= manhattan_distance(robot, env.charge_stations[1]) else env.charge_stations[1]
        #dist_to_charge = manhattan_distance(robot, closest_station)
        closest_package = env.packages[0] \
                    if manhattan_distance(robot.position, env.packages[0].position) \
                    + manhattan_distance(env.packages[0].destination, env.packages[0].position)\
                    <= manhattan_distance(robot.position, env.packages[1].position) \
                    + manhattan_distance(env.packages[1].destination, env.packages[1].position) \
                    else env.packages[1] 
                
        closet_charge = env.charge_stations[0] \
                    if manhattan_distance(robot.position, env.charge_stations[0].position) \
                    <= manhattan_distance(robot.position, env.charge_stations[1].position) \
                    else env.charge_stations[1]
        
        '''path_length = manhattan_distance(robot.position, closest_package.position)
        path_length += manhattan_distance(closest_package.position, closest_package.destination)
        path_length += manhattan_distance(closest_package.destination, closet_charge_to_dest.position)

        charge_length = manhattan_distance(closet_charge.position, closest_package.position)
        charge_length += manhattan_distance(closest_package.position, closest_package.destination)
        charge_length += manhattan_distance(closest_package.destination, closet_charge_to_dest.position)'''


        if env.robot_is_occupied(robot_id):
            next_target = robot.package.destination
        else:
            next_target = closest_package.position

        dist_to_next_target = manhattan_distance(robot.position, next_target)

        closet_charge_to_next = env.charge_stations[0] \
                                if manhattan_distance(next_target, env.charge_stations[0].position) \
                                <= manhattan_distance(next_target, env.charge_stations[1].position) \
                                else env.charge_stations[1]
        
        dist_next_target_to_charge = manhattan_distance(next_target, closet_charge_to_next.position)

        value = 0
        value += robot.credit ** 2
        p = max(min(math.sqrt(max(env.num_steps - robot.credit // 2, 0)), 3), 0)
        print(p)
        value += robot.battery ** p


        if robot.battery < dist_to_next_target + dist_next_target_to_charge and \
            robot.battery >= manhattan_distance(robot.position, closet_charge.position) and \
            robot.battery > env.num_steps // 2 and \
                robot.credit < env.num_steps:
            value -= manhattan_distance(robot.position, closet_charge.position)
        else:
            value -= dist_to_next_target
            if env.robot_is_occupied(robot_id):
                value += manhattan_distance(robot.package.position, robot.package.destination)

        return value
    
    return h(env, robot_id) - h(env, (robot_id + 1) % 2)



    

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators, children = self.successors(env, agent_id)

        start_time = time.time()
        d = 5
        while (time.time() - start_time) * 6 < time_limit:
            best_value = float("-inf")
            best_op = None
            for child, op in zip(children, operators):
                value = self.minimax(child, agent_id, (agent_id + 1) % 2, d)
                if value > best_value:
                    best_value = value
                    best_op = op
            d += 1
        return best_op
           
        
    def minimax(self, env: WarehouseEnv, robot_id, current_robot, D):
        if env.num_steps == 0 or D == 0:
            return smart_heuristic(env, robot_id)

        operators, children = self.successors(env, current_robot)

        if current_robot == robot_id:
            best_value = float("-inf")
            for child in children:
                value = self.minimax(child, robot_id, (current_robot + 1) % 2, D - 1)
                best_value = max(best_value, value)
            return best_value
        else:
            best_value = float("inf")
            for child in children:
                value = self.minimax(child, robot_id, (current_robot + 1) % 2, D - 1)
                best_value = min(best_value, value)
            return best_value


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators, children = self.successors(env, agent_id)

        start_time = time.time()
        d = 5
        while (time.time() - start_time) * 6 < time_limit:
            best_value = float("-inf")
            best_op = None
            alpha = float("-inf")
            beta = float("inf")
            for child, op in zip(children, operators):
                value = self.alphabeta(child, agent_id, (agent_id + 1) % 2, d, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_op = op
                alpha = max(alpha, best_value)
            d += 1
        return best_op
           
        
    def alphabeta(self, env: WarehouseEnv, robot_id, current_robot, D, alpha, beta):
        if env.num_steps == 0 or D == 0:
            return smart_heuristic(env, robot_id)

        operators, children = self.successors(env, current_robot)

        if current_robot == robot_id:
            best_value = float("-inf")
            for child in children:
                value = self.alphabeta(child, robot_id, (current_robot + 1) % 2, D - 1, alpha, beta)
                best_value = max(best_value, value)
                alpha = max(alpha, best_value)
                if beta <= best_value:
                    return float("inf")
            return best_value
        else:
            best_value = float("inf")
            for child in children:
                value = self.alphabeta(child, robot_id, (current_robot + 1) % 2, D - 1, alpha, beta)
                best_value = min(best_value, value)
                beta = min(beta, best_value)
                if alpha >= best_value:
                    return float("-inf")
            return best_value

class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators, children = self.successors(env, agent_id)

        start_time = time.time()
        d = 5
        while (time.time() - start_time) * 6 < time_limit:
            best_value = float("-inf")
            best_op = None
            for child, op in zip(children, operators):
                value = self.expectimax(child, agent_id, (agent_id + 1) % 2, d)
                if value > best_value:
                    best_value = value
                    best_op = op
            d += 1
        return best_op
           
        
    def expectimax(self, env: WarehouseEnv, robot_id, current_robot, D):
        if env.num_steps == 0 or D == 0:
            return smart_heuristic(env, robot_id)

        operators, children = self.successors(env, current_robot)

        num_options = len(operators)
        if "move east" in operators:
            num_options += 1
        if "pick up" in operators:
            num_options += 1

        if current_robot == robot_id:
            best_value = float("-inf")
            for child in children:
                value = self.expectimax(child, robot_id, (current_robot + 1) % 2, D - 1)
                best_value = max(best_value, value)
            return best_value
        else:
            sum = 0
            for op, child in zip(operators, children):
                p = 1 / num_options
                if op == "move east" or op == "pick up":
                    p *= 2
                sum +=  p * self.expectimax(child, robot_id, (current_robot + 1) % 2, D - 1)
            return sum


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)