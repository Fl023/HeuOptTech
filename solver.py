import numpy as np
from collections import defaultdict, deque
import random


class MWCCPSolver:
    def __init__(self, filename, edge_format='adj_list'):
        """
        reading the instance file and initializing
        """
        self.num_u, self.num_v, self.constraints, self.constraints_list, self.edges, self.edge_list = self.read_instance(filename, edge_format)
        self.permutation = [i for i in range(self.num_u + 1, self.num_u + self.num_v + 1)]
        self.solution = None
        self.best_solution = None

    def read_instance(self, filename, edge_format='adj_list'):

        with open(filename, 'r') as file:
            lines = file.readlines()

        first_line = lines[0].strip().split()
        num_u = int(first_line[0])
        num_v = int(first_line[1])
        num_constraints = int(first_line[2])
        num_edges = int(first_line[3])

        constraints_start = lines.index("#constraints\n") + 1
        edges_start = lines.index("#edges\n") + 1

        constraints = defaultdict(list)  # Directed Acyclic Graph
        constraints_list = []
        for line in lines[constraints_start:edges_start - 1]:
            v, v_prime = map(int, line.strip().split())
            constraints[v].append(v_prime)
            constraints_list.append((v, v_prime))

        # Read edges
        edge_list = []
        if edge_format == 'adj_list':
            edges = defaultdict(list)
            for line in lines[edges_start:]:
                u, v, weight = line.strip().split()
                edges[int(u)].append((int(v), float(weight)))
                edge_list.append((int(u), int(v), float(weight)))
        elif edge_format == 'matrix':
            edges = np.zeros((num_u, num_v))
            for line in lines[edges_start:]:
                u, v, weight = line.strip().split()
                edges[int(u) - 1, int(v) - num_u - 1] = float(weight)
                edge_list.append((int(u), int(v), float(weight)))
        else:
            raise ValueError("Invalid edge_format. Use 'adj_list' or 'matrix'.")

        return num_u, num_v, constraints, constraints_list, edges, edge_list

    def calculate_objective(self, permutation):
        """
        total weighted crossings for a given permutation
        """
        pos = {node: i for i, node in enumerate(permutation)}
        total_crossings = 0

        for i in range(len(self.edge_list)):
            u, v, w = self.edge_list[i]
            for j in range(len(self.edge_list)):
                u_prime, v_prime, w_prime = self.edge_list[j]
                if u < u_prime:
                    if pos[v] > pos[v_prime]:
                        total_crossings += w + w_prime

        return total_crossings

    def is_feasible(self, permutation):
        """
        check if given permutation satisfies all constraints
        """
        position = {node: i for i, node in enumerate(permutation)}
        for v, v_prime in self.constraints_list:
            if position[v] >= position[v_prime]:
                return False
        return True

    def swap_neighbors(self, solution):
        """
        swapping adjacent nodes
        """
        neighbors = []
        for i in range(self.num_v - 1):
            neighbor = solution[:]
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
            if self.is_feasible(neighbor):
                neighbors.append(neighbor)

        neighbor = self.permutation[:]
        neighbor[0], neighbor[-1] = neighbor[-1], neighbor[0]
        if self.is_feasible(neighbor):
            neighbors.append(neighbor)

        return neighbors

    def insert_neighbors(self, solution):
        """
        inserting a node to a different position
        """
        neighbors = []
        for i in range(self.num_v):
            for j in range(self.num_v):
                if i != j:
                    neighbor = solution[:]
                    node = neighbor.pop(i)
                    neighbor.insert(j, node)
                    if self.is_feasible(neighbor):
                        neighbors.append(neighbor)
        return neighbors

    def reverse_segment_cyclic(self, solution, segment_length=2):
        """
        reversing a fixed-length segment cyclically
        """
        neighbors = []
        n = self.num_v

        for start in range(n):
            neighbor = solution[:]
            segment = [neighbor[(start + i) % n] for i in range(segment_length)]
            for i in range(segment_length):
                neighbor[(start + i) % n] = segment[segment_length - 1 - i]
            if self.is_feasible(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def reverse_segment_fixed(self, solution, segment_length=2):
        """
        reversing fixed-length segments
        """
        neighbors = []
        n = self.num_v

        for i in range(n - segment_length + 1):
            neighbor = solution[:]
            neighbor[i:i + segment_length] = reversed(neighbor[i:i + segment_length])
            if self.is_feasible(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def reverse_segment_all_sizes(self, solution):
        """
        reversing segments of all possible sizes
        """
        neighbors = []
        n = self.num_v

        for i in range(n):
            for j in range(i + 1, n):
                neighbor = solution[:]
                neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
                if self.is_feasible(neighbor):
                    neighbors.append(neighbor)

        return neighbors

    def topological_sort(self):
        """
        Kahn’s algorithm for Topological Sorting
        """
        in_degree = {node: 0 for node in self.permutation}
        graph = defaultdict(list)

        for v, v_prime in self.constraints_list:
            graph[v].append(v_prime)
            in_degree[v_prime] += 1

        zero_in_degree = deque([node for node in self.permutation if in_degree[node] == 0])
        sorted_order = []

        while zero_in_degree:
            current = zero_in_degree.popleft()
            sorted_order.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

        return sorted_order

    def deterministic_construction_heuristic(self):
        """
        needs something better
        """
        return self.topological_sort()

    def randomized_construction_heuristic(self):
        """
        ---
        """
        return self.topological_sort()

    def first_improvement(self, f_x, neighbors):
        """
        select the first neighbor that improves the solution
        """
        for neighbor in neighbors:
            neighbor_value = self.calculate_objective(neighbor)
            if neighbor_value < f_x:  # Look for improvement
                return neighbor, neighbor_value
        return None, f_x  # No improvement found

    def best_improvement(self, f_x, neighbors):
        """
        select the best neighbor that improves the solution.
        """
        best_neighbor = None
        best_value = f_x

        for neighbor in neighbors:
            neighbor_value = self.calculate_objective(neighbor)
            if neighbor_value < best_value:  # Look for the best improvement
                best_neighbor = neighbor
                best_value = neighbor_value

        return best_neighbor, best_value  # Return the best neighbor (or None if no improvement is found)

    def random_step(self, f_x, neighbors):
        """
        select a random neighbor.
        """
        random.seed(0)
        if neighbors:
            rand_neighbor = random.choice(neighbors)
            return rand_neighbor, self.calculate_objective(rand_neighbor)
        return None, f_x  # Return None if there are no neighbors

    def local_search(self, neighbors_func, step_function, max_iterations=100):
        """
        local search
        """
        current_solution = self.solution
        for _ in range(max_iterations):
            f = self.calculate_objective(current_solution)
            neighbors = neighbors_func(current_solution)
            next_solution, f_prime = step_function(f, neighbors)
            if next_solution is None:
                break
            if f_prime <= f:
                current_solution = next_solution
            else:
                break
        return current_solution

    def grasp(self, neighbors_func, step_function, max_iterations=100):

        best_solution = None
        best_objective = float('inf')

        for iteration in range(max_iterations):
            self.solution = self.randomized_construction_heuristic()
            solution = self.local_search(neighbors_func, step_function, max_iterations=max_iterations)
            objective_value = self.calculate_objective(solution)
            if objective_value < best_objective:
                best_solution = solution
                best_objective = objective_value

        return best_solution

    def solve_local_search(self, step_function='best_improvement', neighbors_func='swap_neighbors', segment_length=None):

        self.solution = self.topological_sort()
        print("step function:", step_function)
        print("neighborhood:", neighbors_func)
        print("segment length:", segment_length)
        print("Initial solution:", self.solution)

        neighbors_func_map = {
            'swap_neighbors': self.swap_neighbors,
            'insert_neighbors': self.insert_neighbors,
            'reverse_segment_cyclic': lambda solution: self.reverse_segment_cyclic(solution, segment_length),
            'reverse_segment_fixed': lambda solution: self.reverse_segment_fixed(solution, segment_length),
            'reverse_segment_all_sizes': self.reverse_segment_all_sizes,
        }

        step_func_map = {
            'best_improvement': self.best_improvement,
            'first_improvement': self.first_improvement,
            'random': self.random_step
        }

        neighbors_func = neighbors_func_map[neighbors_func]
        step_func = step_func_map[step_function]

        self.best_solution = self.local_search(neighbors_func, step_func, max_iterations=100)
        print("Best solution:", self.best_solution)

        f1 = self.calculate_objective(self.solution)
        f_best = self.calculate_objective(self.best_solution)

        print("Initial objective value:", f1)
        print("Best objective value:", f_best)


solver = MWCCPSolver("test_instances/small/inst_50_4_00010")
solver.solve_local_search(step_function='first_improvement', neighbors_func='reverse_segment_all_sizes', segment_length=2)
