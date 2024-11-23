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

    def calculate_objective(self, permutation):    # FASTER!

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

    def calculate_partial_objective(self, permutation):    # FASTER!

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
        return neighbors

    def swap_neighbors_delta(self, solution, step):
        """
        swapping adjacent nodes
        """
        neighbors = []
        deltas = []
        indices = []
        for i in range(self.num_v - 1):
            neighbor = solution[:]
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
            if self.is_feasible(neighbor):
                neighbors.append(neighbor)
                indices.append(i)
                if step == 'first_improvement':
                    delta = self.delta_obj_swap(neighbor[i], neighbor[i + 1])
                    if delta < 0:
                        return neighbor, delta
                if step == 'best_improvement':
                    delta = self.delta_obj_swap(neighbor[i], neighbor[i + 1])
                    deltas.append(delta)

        if step == 'best_improvement' and neighbors:
            index_best = deltas.index(min(deltas))
            if deltas[index_best] < 0:
                return neighbors[index_best], deltas[index_best]
        if step == 'random' and neighbors:
            random_index = random.randrange(len(neighbors))
            random_neighbor = neighbors[random_index]
            delta = self.delta_obj_swap(random_neighbor[indices[random_index]],
                                        random_neighbor[indices[random_index] + 1])
            return random_neighbor, delta

        return None, 0

    def delta_obj_swap(self, node1, node2):
        new = self.calculate_partial_objective([node1, node2])
        old = self.calculate_partial_objective([node2, node1])
        return new - old

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

    def insert_neighbors_delta(self, solution, step):

        neighbors = []
        deltas = []
        indices = []
        for i in range(self.num_v):
            for j in range(self.num_v):
                if i != j:
                    neighbor = solution[:]
                    node = neighbor.pop(i)
                    neighbor.insert(j, node)
                    if self.is_feasible(neighbor):
                        neighbors.append(neighbor)
                        indices.append((i,j))
                        if i < j:
                            subsolution = solution[i:j + 1]
                            subneighbor = neighbor[i:j + 1]
                        else:
                            subsolution = solution[j:i + 1]
                            subneighbor = neighbor[j:i + 1]
                        if step == 'first_improvement':
                            delta = self.calculate_partial_objective(subneighbor) - \
                                    self.calculate_partial_objective(subsolution)
                            if delta < 0:
                                return neighbor, delta
                        if step == 'best_improvement':
                            delta = self.calculate_partial_objective(subneighbor) - \
                                    self.calculate_partial_objective(subsolution)
                            deltas.append(delta)

        if step == 'best_improvement' and neighbors:
            index_best = deltas.index(min(deltas))
            if deltas[index_best] < 0:
                return neighbors[index_best], deltas[index_best]
        if step == 'random' and neighbors:
            random_index = random.randrange(len(neighbors))
            random_neighbor = neighbors[random_index]
            i, j = indices[random_index]
            if i < j:
                subsolution = solution[i:j + 1]
                subneighbor = random_neighbor[i:j + 1]
            else:
                subsolution = solution[j:i + 1]
                subneighbor = random_neighbor[j:i + 1]
            delta = self.calculate_partial_objective(subneighbor) - \
                    self.calculate_partial_objective(subsolution)
            return random_neighbor, delta

        return None, 0

    def reverse_segment(self, solution, segment_length=2):
        """
        reversing segments of all possible sizes between 2 and segment_length
        """
        neighbors = []
        n = self.num_v

        for i in range(n):
            end = i+segment_length
            if end < n+1:
                for j in range(i + 1, end):
                    neighbor = solution[:]
                    neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
                    if self.is_feasible(neighbor):
                        neighbors.append(neighbor)

        return neighbors

    def reverse_segment_delta(self, solution, step, segment_length=5):
        """
        reversing segments of all possible sizes between 2 and segment_length
        """
        neighbors = []
        deltas = []
        indices = []
        n = self.num_v

        for i in range(n):
            end = i+segment_length
            if end < n+1:
                for j in range(i + 1, end):
                    neighbor = solution[:]
                    neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
                    if self.is_feasible(neighbor):
                        neighbors.append(neighbor)
                        indices.append((i, j))
                        subsolution = solution[i:j + 1]
                        subneighbor = neighbor[i:j + 1]
                        if step == 'first_improvement':
                            delta = self.calculate_partial_objective(subneighbor) - \
                                    self.calculate_partial_objective(subsolution)
                            if delta < 0:
                                return neighbor, delta
                        if step == 'best_improvement':
                            delta = self.calculate_partial_objective(subneighbor) - \
                                    self.calculate_partial_objective(subsolution)
                            deltas.append(delta)

        if step == 'best_improvement' and neighbors:
            index_best = deltas.index(min(deltas))
            if deltas[index_best] < 0:
                return neighbors[index_best], deltas[index_best]
        if step == 'random' and neighbors:
            random_index = random.randrange(len(neighbors))
            random_neighbor = neighbors[random_index]
            i, j = indices[random_index]
            subsolution = solution[i:j + 1]
            subneighbor = random_neighbor[i:j + 1]
            delta = self.calculate_partial_objective(subneighbor) - \
                    self.calculate_partial_objective(subsolution)
            return random_neighbor, delta

        return None, 0

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
                weighted_crossings.append((candidate, self.calculate_partial_objective(solution + [candidate])))

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
        Randomized construction heuristic.
        Works like topological sort but randomly chooses the next node to add
        from the list of restricted nodes with in_degree = 0, ensuring feasibility.
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
            candidate_weights = {candidate: self.calculate_partial_objective(solution + [candidate])
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
        f = self.calculate_objective(current_solution)
        for _ in range(max_iterations):
            neighbors = neighbors_func(current_solution)
            next_solution, f_prime = step_function(f, neighbors)
            if next_solution is None:
                break
            if f_prime <= f:
                current_solution = next_solution
                f = f_prime
            elif step_function != self.random_step:
                break
        return current_solution

    def local_search_delta(self, init_solution, neighbors_func, step_function, max_iterations=100):
        """
        local search
        """
        current_solution = init_solution
        f = self.calculate_objective(current_solution)
        for _ in range(max_iterations):
            neighbor, delta = neighbors_func(current_solution, step_function)
            if neighbor is None:
                break
            if delta <= 0:
                f += delta
                current_solution = neighbor
            elif step_function != 'random':
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

        init_solution = self.topological_sort()
        # init_solution = self.deterministic_construction_heuristic()
        print("step function:", step_function)
        print("neighborhood:", neighbors_func)
        print("segment length:", segment_length)
        print("Initial solution:", init_solution)

        neighbors_func_map = {
            'swap_neighbors': self.swap_neighbors,
            'insert_neighbors': self.insert_neighbors,
            'reverse_segment': lambda solution: self.reverse_segment(solution, segment_length),
        }

        step_func_map = {
            'best_improvement': self.best_improvement,
            'first_improvement': self.first_improvement,
            'random': self.random_step
        }

        neighbors_func = neighbors_func_map[neighbors_func]
        step_func = step_func_map[step_function]

        start_time = time.time()
        best_solution = self.local_search(init_solution, neighbors_func, step_func, max_iterations=100)
        end_time = time.time()
        print("runtime:", end_time - start_time)
        print("Best solution:", best_solution)

        f1 = self.calculate_objective(init_solution)
        f_best = self.calculate_objective(best_solution)

        print("Initial objective value:", f1)
        print("Best objective value:", f_best)

    def solve_local_search_delta(self, step_function='best_improvement', neighbors_func='swap_neighbors', segment_length=None):

        init_solution = self.topological_sort()
        # init_solution = self.deterministic_construction_heuristic()
        print("step function:", step_function)
        print("neighborhood:", neighbors_func)
        print("segment length:", segment_length)
        print("Initial solution:", init_solution)

        neighbors_func_map = {
            'swap_neighbors': self.swap_neighbors_delta,
            'insert_neighbors': self.insert_neighbors_delta,
            'reverse_segment': lambda solution, step: self.reverse_segment_delta(solution, step, segment_length),
        }

        neighbors_func = neighbors_func_map[neighbors_func]

        start_time = time.time()
        best_solution = self.local_search_delta(init_solution, neighbors_func, step_function, max_iterations=100)
        end_time = time.time()
        print("runtime:", end_time - start_time)
        print("Best solution:", best_solution)

        f1 = self.calculate_objective(init_solution)
        f_best = self.calculate_objective(best_solution)

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
            'reverse_segment': lambda solution: self.reverse_segment(solution, segment_length),
        }

        step_func_map = {
            'best_improvement': self.best_improvement,
            'first_improvement': self.first_improvement,
            'random': self.random_step
        }

        neighbors_func = neighbors_func_map[neighbors_func]
        step_func = step_func_map[step_function]

        best_solution = self.grasp(neighbors_func, step_func, alpha=0.2, max_iterations=max_iterations)
        print("Best solution:", best_solution)

        f1 = self.calculate_objective(init_solution)
        f_best = self.calculate_objective(best_solution)

        print("Initial objective value:", f1)
        print("Best objective value:", f_best)

    def testing(self):
        self.solution = self.topological_sort()
        # self.solution = self.deterministic_construction_heuristic()
        start_time = time.time()
        sol = self.local_search_delta(self.solution, self.reverse_segment_delta, 'random', max_iterations=100)
        end_time = time.time()
        obj_val = self.calculate_objective(self.solution)
        obj_val_sol = self.calculate_objective(sol)

        # print(self.solution, obj_val)
        print(sol, obj_val_sol)
        print("runtime:", end_time-start_time)






solver = MWCCPSolver("test_instances/small/inst_50_4_00010", seed=1)
solver.solve_local_search(step_function='best_improvement', neighbors_func='reverse_segment', segment_length=5)
solver.solve_local_search_delta(step_function='best_improvement', neighbors_func='reverse_segment', segment_length=5)
# solver.solve_grasp(step_function='best_improvement', neighbors_func='reverse_segment',
#                    segment_length=2, alpha=0.2, max_iterations=30)

