import numpy as np
from collections import defaultdict, deque
import random
import time


class MWCCPSolver:
    def __init__(self, filename, edge_format='adj_list', seed=None):
        """
        reading the instance file and initializing
        """
        self.num_u, self.num_v, self.constraints, self.constraints_list, self.edges, self.edge_list = self.read_instance(filename, edge_format)
        self.permutation = [i for i in range(self.num_u + 1, self.num_u + self.num_v + 1)]
        self.solution = None
        self.best_solution = None

        if seed is not None:
            random.seed(seed)
            print(f"Random seed set to: {seed}")

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
                edges[int(v)].append((int(u), float(weight)))
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

    def calculate_objective(self, permutation):     # SLOW!
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

    def calculate_objective2(self, permutation):    # FASTER!

        # Create a mapping of node positions based on the permutation
        pos = {node: i for i, node in enumerate(permutation)}
        total_crossings = 0

        for v, us in self.edges.items():
            for v_prime, us_prime in self.edges.items():
                if pos[v] > pos[v_prime]:
                    for u, w in us:
                        for u_prime, w_prime in us_prime:
                            if u < u_prime:
                                total_crossings += w + w_prime

        return total_crossings

    def calculate_partial_objective(self, permutation):     # SLOW!
        """
        total weighted crossings for a given permutation
        """
        pos = {node: i for i, node in enumerate(permutation)}
        total_crossings = 0
        partial_edges = [edge for edge in self.edge_list if edge[1] in permutation]

        for i in range(len(partial_edges)):
            u, v, w = partial_edges[i]
            for j in range(len(partial_edges)):
                u_prime, v_prime, w_prime = partial_edges[j]
                if u < u_prime:
                    if pos[v] > pos[v_prime]:
                        total_crossings += w + w_prime

        return total_crossings

    def calculate_partial_objective2(self, permutation):    # FASTER!

        # Create a mapping of node positions based on the permutation
        pos = {node: i for i, node in enumerate(permutation)}
        partial_edges = {key: self.edges[key] for key in permutation}
        total_crossings = 0

        for v, us in partial_edges.items():
            for v_prime, us_prime in partial_edges.items():
                if pos[v] > pos[v_prime]:
                    for u, w in us:
                        for u_prime, w_prime in us_prime:
                            if u < u_prime:
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

        # # alternative:
        # for v in self.constraints:
        #     for v_prime in self.constraints[v]:
        #         if position[v] >= position[v_prime]:
        #             return False
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
        works like topological sort, but chooses node from list of nodes of in_degree=0,
        which leads to the smallest increase in total weight
        """
        # Initialize
        solution = []  # Start with an empty solution
        in_degree = {node: 0 for node in self.permutation}
        graph = defaultdict(list)
        # outgoing_weights = defaultdict(float)  # Store total outgoing weights for each node
        outgoing_weights = dict.fromkeys(self.permutation, 0.0)

        # Build the graph, in-degree, and outgoing weights based on constraints and edges
        for v, v_prime in self.constraints_list:
            graph[v].append(v_prime)
            in_degree[v_prime] += 1

        for u, v, w in self.edge_list:
            outgoing_weights[v] += w

        # Initialize the candidate list with nodes that have zero in-degree
        candidate_list = [node for node in self.permutation if in_degree[node] == 0]
        # Step 1: Select the first node based on total outgoing edge weight
        first_node = min(candidate_list, key=lambda x: outgoing_weights[x])
        solution.append(first_node)
        candidate_list.remove(first_node)

        # Update in-degree and candidate list after adding the first node
        for neighbor in graph[first_node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                candidate_list.append(neighbor)

        # Iteratively construct the rest of the solution
        while candidate_list:
            # Calculate weighted crossings for each candidate
            weighted_crossings = []
            for candidate in candidate_list:
                weighted_crossings.append((candidate, self.calculate_partial_objective2(solution + [candidate])))

            # Sort candidates by weighted crossings (ascending)
            weighted_crossings.sort(key=lambda x: x[1])     # sort not nessecary, just take min

            # Select the candidate with the least weighted crossings
            best_candidate = weighted_crossings[0][0]

            # Add the selected node to the solution
            solution.append(best_candidate)
            candidate_list.remove(best_candidate)

            # Update in-degree and candidate list
            for neighbor in graph[best_candidate]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    candidate_list.append(neighbor)

        return solution

    def random_construction_with_rcl(self, alpha=0.5):
        """
        Randomized construction heuristic with Restricted Candidate List (RCL).
        Works like topological sort but selects the next node randomly from the RCL,
        ensuring feasibility and balancing greediness and randomness.

        Parameters:
            alpha (float): Parameter controlling the greediness (0 = purely greedy, 1 = purely random).

        Returns:
            solution (list): Constructed solution.
        """
        # Initialize
        solution = []  # Start with an empty solution
        in_degree = {node: 0 for node in self.permutation}
        graph = defaultdict(list)
        outgoing_weights = dict.fromkeys(self.permutation, 0.0)  # Store total outgoing weights for each node

        # Build the graph, in-degree, and outgoing weights based on constraints and edges
        for v, v_prime in self.constraints_list:
            graph[v].append(v_prime)
            in_degree[v_prime] += 1

        for u, v, w in self.edge_list:
            outgoing_weights[v] += w  # Correct: should be v

        # Initialize the candidate list with nodes that have zero in-degree
        candidate_list = [node for node in self.permutation if in_degree[node] == 0]

        c_min = min(outgoing_weights[node] for node in candidate_list)
        c_max = max(outgoing_weights[node] for node in candidate_list)

        # Compute the RCL threshold
        threshold = c_min + alpha * (c_max - c_min)

        # Build the RCL with nodes satisfying the threshold condition
        rcl = [node for node in candidate_list if outgoing_weights[node] <= threshold]

        # Randomly select the first node from the RCL
        first_node = random.choice(rcl)
        solution.append(first_node)
        candidate_list.remove(first_node)

        # Update in-degree and candidate list after adding the first node
        for neighbor in graph[first_node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                candidate_list.append(neighbor)

        # Iteratively construct the solution
        while candidate_list:
            # Compute the weights (c(e)) for all candidates
            candidate_weights = {candidate: self.calculate_partial_objective2(solution + [candidate])
                                 for candidate in candidate_list}

            # Determine c_min and c_max
            c_min = min(candidate_weights.values())
            c_max = max(candidate_weights.values())

            # Compute the RCL threshold
            threshold = c_min + alpha * (c_max - c_min)

            # Build the Restricted Candidate List (RCL)
            rcl = [candidate for candidate, weight in candidate_weights.items() if weight <= threshold]

            # Randomly select a node from the RCL
            selected_candidate = random.choice(rcl)

            # Add the selected node to the solution
            solution.append(selected_candidate)
            candidate_list.remove(selected_candidate)

            # Update in-degree and candidate list
            for neighbor in graph[selected_candidate]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    candidate_list.append(neighbor)

        return solution

    def random_construction_heuristic(self):
        """
        Randomized construction heuristic.
        Works like topological sort but randomly chooses the next node to add
        from the list of nodes with in_degree = 0, ensuring feasibility.
        """
        # Initialize
        solution = []  # Start with an empty solution
        in_degree = {node: 0 for node in self.permutation}
        graph = defaultdict(list)
        outgoing_weights = defaultdict(float)  # Store total outgoing weights for each node

        # Build the graph, in-degree, and outgoing weights based on constraints and edges
        for v, v_prime in self.constraints_list:
            graph[v].append(v_prime)
            in_degree[v_prime] += 1

        for u, v, w in self.edge_list:
            outgoing_weights[v] += w  # Correct: should be v

        # Initialize the candidate list with nodes that have zero in-degree
        candidate_list = [node for node in self.permutation if in_degree[node] == 0]

        # Select the first node randomly from the candidates
        first_node = random.choice(candidate_list)
        solution.append(first_node)
        candidate_list.remove(first_node)

        # Update in-degree and candidate list after adding the first node
        for neighbor in graph[first_node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                candidate_list.append(neighbor)

        # Iteratively construct the rest of the solution
        while candidate_list:
            # Randomly select a node from the candidate list
            selected_candidate = random.choice(candidate_list)

            # Add the selected node to the solution
            solution.append(selected_candidate)
            candidate_list.remove(selected_candidate)

            # Update in-degree and candidate list
            for neighbor in graph[selected_candidate]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    candidate_list.append(neighbor)

        return solution

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
        if neighbors:
            rand_neighbor = random.choice(neighbors)
            return rand_neighbor, self.calculate_objective(rand_neighbor)
        return None, f_x  # Return None if there are no neighbors

    def local_search(self, init_solution, neighbors_func, step_function, max_iterations=100):
        """
        local search
        """
        current_solution = init_solution
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

    def grasp(self, neighbors_func, step_function, alpha=0.2, max_iterations=100):

        best_solution = None
        best_objective = float('inf')

        for iteration in range(max_iterations):
            x = self.random_construction_with_rcl(alpha=alpha)
            x_prime = self.local_search(x, neighbors_func, step_function)
            objective_value = self.calculate_objective(x_prime)
            if objective_value < best_objective:
                best_solution = x_prime
                best_objective = objective_value

        return best_solution

    def solve_local_search(self, step_function='best_improvement', neighbors_func='swap_neighbors', segment_length=None):

        # init_solution = self.topological_sort()
        init_solution = self.deterministic_construction_heuristic()
        print("step function:", step_function)
        print("neighborhood:", neighbors_func)
        print("segment length:", segment_length)
        print("Initial solution:", init_solution)

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

        self.best_solution = self.local_search(init_solution, neighbors_func, step_func, max_iterations=100)
        print("Best solution:", self.best_solution)

        f1 = self.calculate_objective(init_solution)
        f_best = self.calculate_objective(self.best_solution)

        print("Initial objective value:", f1)
        print("Best objective value:", f_best)

    def solve_grasp(self, step_function='best_improvement', neighbors_func='swap_neighbors',
                    segment_length=None, alpha=0.2, max_iterations=100):

        init_solution = self.deterministic_construction_heuristic()
        print("step function:", step_function)
        print("neighborhood:", neighbors_func)
        print("segment length:", segment_length)
        print("Initial solution:", init_solution)

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

        self.best_solution = self.grasp(neighbors_func, step_func, alpha=0.2, max_iterations=max_iterations)
        print("Best solution:", self.best_solution)

        f1 = self.calculate_objective(init_solution)
        f_best = self.calculate_objective(self.best_solution)

        print("Initial objective value:", f1)
        print("Best objective value:", f_best)

    def testing(self):
        self.solution = self.topological_sort()
        # print("deterministic:    ", self.deterministic_construction_heuristic())
        print("completely random:", self.random_construction_heuristic())
        # print("alpha=0.5:        ", self.random_construction_with_rcl(alpha=0.5))
        # print("alpha=0:          ", self.random_construction_with_rcl(alpha=0.0))
        # print("alpha=1:          ", self.random_construction_with_rcl(alpha=1.0))
        # print(self.solution)
        # dch = self.deterministic_construction_heuristic()
        # print(dch)
        # partial = []
        # for node in self.solution:
        #     partial.append(node)
        #     print(self.calculate_partial_objective2(partial))
        # print(self.calculate_partial_objective2(self.solution))
        # print(self.calculate_partial_objective2(dch))

        # for i in range(10):
        #
        #     start_time = time.time()
        #     original = self.calculate_objective(self.solution)
        #     end_time = time.time()
        #     elapsed_time = (end_time - start_time)*1000
        #     print(f"original Function executed in {elapsed_time:.6f} ms")
        #     print("original:", original)
        #
        #     start_time = time.time()
        #     first = self.calculate_partial_objective(self.solution)
        #     end_time = time.time()
        #     elapsed_time = (end_time - start_time) * 1000
        #     print(f"first Function executed in {elapsed_time:.6f} ms")
        #     print("first:", first)
        #
        #     start_time = time.time()
        #     second = self.calculate_partial_objective2(self.solution)
        #     end_time = time.time()
        #     elapsed_time = (end_time - start_time) * 1000
        #     print(f"second Function executed in {elapsed_time:.6f} ms")
        #     print("second:", second)
        #
        #     start_time = time.time()
        #     third = self.calculate_objective2(self.solution)
        #     end_time = time.time()
        #     elapsed_time = (end_time - start_time) * 1000
        #     print(f"third Function executed in {elapsed_time:.6f} ms")
        #     print("third:", third)





solver = MWCCPSolver("test_instances/small/inst_50_4_00010", seed=0)
# solver.testing()
# solver.solve_local_search(step_function='first_improvement', neighbors_func='swap_neighbors', segment_length=2)
solver.solve_grasp(step_function='first_improvement', neighbors_func='swap_neighbors',
                   segment_length=2, alpha=0.2, max_iterations=30)

