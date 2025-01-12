import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt

##############################################################################
#                           PROFILING DECORATOR
##############################################################################

profile_data = {}  # holds cumulative runtime of each function

def profile(func):
    """
    Decorator that measures how long 'func' takes to run,
    accumulates total time in 'profile_data[func.__name__]'.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        profile_data[func.__name__] = profile_data.get(func.__name__, 0) + elapsed
        return result
    return wrapper


##############################################################################
#                           CORE ACO CODE
##############################################################################

@profile
def read_instance(filename):
    """
    Reads:
      - First line: num_u, num_v
      - #constraints\n => lines with 'v, v_prime' => v < v_prime
      - #edges\n => lines with 'u, v, weight' (u in top, v in bottom)

    Returns (num_u, num_v, constraints, edgesdic).
    'constraints[a]' = set of b => a < b.
    'edgesdic[v]' = list of (top_node, weight) for crossing calc.
    """
    #print(f"[INFO] Reading instance: {filename}")
    with open(filename, 'r') as f:
        lines = f.readlines()

    first_line = lines[0].strip().split()
    num_u, num_v = int(first_line[0]), int(first_line[1])

    constraints_start = lines.index("#constraints\n") + 1
    edges_start       = lines.index("#edges\n") + 1

    constraints = defaultdict(set)
    for line in lines[constraints_start : edges_start - 1]:
        v, v_prime = map(int, line.strip().split())
        # "v < v_prime"
        # shift bottom nodes => 0..(num_v - 1)
        constraints[v - num_u - 1].add(v_prime - num_u - 1)

    edgesdic = defaultdict(list)
    for line in lines[edges_start:]:
        u_str, v_str, w_str = line.strip().split()
        u, v = int(u_str), int(v_str)
        w = float(w_str)
        # shift top => (u-1), bottom => (v-num_u-1)
        edgesdic[v - num_u - 1].append((u - 1, w))

    edgesdicnumpy = defaultdict(lambda: np.array([[]]))
    for key, value in edgesdic.items():
        edgesdicnumpy[key] = np.array(value)

    return num_u, num_v, constraints, edgesdicnumpy

@profile
def propagate_constraints(constraints, num_v):
    """
    Ensures transitive closure:
    If A < B and B < C, then A < C.
    We'll do a simple Floyd-Warshall or BFS approach on a boolean adjacency matrix.
    """
    adjacency = np.zeros((num_v, num_v), dtype=bool)
    for a in constraints:
        for b in constraints[a]:
            adjacency[a, b] = True

    # Floyd-Warshall-like approach
    for k in range(num_v):
        for i in range(num_v):
            if adjacency[i, k]:
                for j in range(num_v):
                    if adjacency[k, j]:
                        adjacency[i, j] = True

    # Rebuild constraints from adjacency
    new_constraints = defaultdict(set)
    for i in range(num_v):
        for j in range(num_v):
            if adjacency[i, j]:
                new_constraints[i].add(j)
    return new_constraints

@profile
def compute_predecessors(constraints, num_v):
    """
    Build a list/dict of predecessors: for each node n, we want the set of all 'a'
    such that a < n in the constraints. 
    This is basically the inverse of constraints[a] = set(b) => a < b,
    so we gather for each 'b', all 'a' that must precede 'b'.
    """
    predecessors = [set() for _ in range(num_v)]
    # If a < b => b has a as a predecessor
    for a in constraints:
        for b in constraints[a]:
            # b cannot appear until a is placed
            predecessors[b].add(a)
    return predecessors

@profile
def build_crossing_vals(num_v, edgesdic):
    """
    crossing_vals[i][j] = cost if node i is placed immediately before node j.
    """
    cost_count = 0
    cost_sum = 0

    nodes = np.arange(0, num_v, dtype=int)
    cross_vals = np.zeros((num_v, num_v))
    for i in nodes:
        edges_prime = edgesdic[i]
        if edges_prime.size == 0:
            continue
        u2, w2 = edges_prime[:, 0], edges_prime[:, 1]
        for j in nodes:
            if i == j:
                continue
            edges = edgesdic[j]
            if edges.size == 0:
                continue
            u1, w1 = edges[:, 0], edges[:, 1]
            total_crossings = np.sum((w1[:, None] + w2) * (u1[:, None] < u2))
            cross_vals[i][j] = total_crossings

            cost_sum += total_crossings
            cost_count += 1

    average_cost = cost_sum/cost_count
    return cross_vals, average_cost

@profile
def is_feasible(perm, constraints):
    """
    Checks if 'perm' array of nodes satisfies constraints (a < b).
    constraints[a] = set of b => a < b.
    """
    position = np.empty(len(perm), dtype=int)
    for i, node in enumerate(perm):
        position[node] = i

    for a in constraints:
        for b in constraints[a]:
            if position[a] >= position[b]:
                return False
    return True

@profile
def consecutive_cost(perm, crossing_vals):
    """
    Sums crossing_vals[a,b] for consecutive pairs (a,b) in 'perm'.
    """
    total = 0.0
    for i in range(len(perm) - 1):
        a = perm[i]
        b = perm[i+1]
        total += crossing_vals[a, b]
    return total


@profile
def build_static_feasible_lists(num_v, constraints):
    """
    Returns a list of lists `feasible_after[i]`, where `feasible_after[i]` is 
    all nodes j that can come immediately after i *if we only consider static 
    ordering constraints*
    """
    not_feasible_before = [set() for _ in range(num_v)]
    
    for k, v in constraints.items():
        for value in v:
            not_feasible_before[value].add(k)
    
    return not_feasible_before

    

@profile
def build_ant_solution(num_v, not_feasible_before, crossing_vals, pheromone, pheromone_start, average_cost, alpha=1.0, beta=1.0):
    """
    Builds a single feasible permutation using discrete step-by-step ACO logic,
    reading precomputed adjacency 'feasible_after[i]' that lists which nodes 
    *can* follow i in principle (static constraints). We still filter out 
    visited nodes to ensure no duplication & partial-solution feasibility.
    
    If no feasible next node is found at some step, returns None.
    
    'feasible_after[i]' -> list of j that can come after i (ignoring partial-solution specifics).
    """
    unvisited = set(range(num_v))
    solution = []
    current_node = -1  # indicates 'start'

    while unvisited:
        # 1) gather potential from precomputed adjacency
        if current_node == -1:
            # if we have a "start" node concept => we can do uniform or do we have feasible_after[-1]? 
            # Typically we do a uniform approach for 'start' => so let's allow all unvisited.
            potential_next = list(unvisited)
        else:
            potential_next = []
            for j in unvisited:
                if j == current_node: 
                    continue
                potential_next.append(j)

        # 2) Pheromone row
        if current_node == -1:
            row = pheromone_start
        else:
            row = pheromone[current_node, :]

        # 3) Heuristic => prefer lower crossing
        if current_node == -1:
            heuristic = np.ones(num_v, dtype=float)
        else:
            cval = crossing_vals[current_node, :]
            heuristic = 3*average_cost / (1.0 + cval)

        # 4) Build scores
        scores = []
        for node_candidate in potential_next:
            ph = row[node_candidate]**alpha
            he = heuristic[node_candidate]**beta
            scores.append(ph * he)

        # 5) Probability distribution
        scores = np.array(scores, dtype=float)
        p = scores / scores.sum()
        while(len(potential_next) > 0):
            if np.isnan(p).any():
                return None
            next_node = np.random.choice(potential_next, p=p)
            for i in not_feasible_before[next_node]:
                if not i in solution:
                    p[potential_next.index(next_node)] = 0
                    try:
                        p = p / p.sum()
                    except ZeroDivisionError:
                        return None
                    next_node = None
                    break
            if next_node != None:
                break
        else:
            print("###########################Weird############################################")

        

        if next_node == None:
            return None

        # 6) update solution
        solution.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node

    return solution

@profile
def ant_colony_opt(
    filename, 
    num_ants=10, 
    alpha=1.0, 
    beta=1.0, 
    rho=0.1, 
    Q=1.0, 
    max_time_sec=10, 
    seed=0
):
    """
    Main ACO driver (time-based stopping):
      - read instance
      - propagate constraints
      - compute_predecessors => for each node, which must be left of it
      - build crossing_vals
      - init pheromone => all ones
      - while time budget not exceeded:
         * ants build solutions
         * pick best local
         * evaporate
         * deposit pheromone
      - keep global best until time runs out
    """
    import time

    np.random.seed(seed)
    local_bests = []

    # 1) read
    num_u, num_v, constraints_dict, edgesdic = read_instance(filename)
    # 2) propagate
    constraints = propagate_constraints(constraints_dict, num_v)
    # 3) compute predecessors
    predecessors = compute_predecessors(constraints, num_v)
    # 4) crossing
    crossing_vals, average_cost = build_crossing_vals(num_v, edgesdic)
    # 5) init pheromone
    pheromone = np.ones((num_v, num_v), dtype=float)
    pheromone_start = np.ones(num_v, dtype=float)

    best_perm = None
    best_cost = float('inf')

    not_feasible_before = build_static_feasible_lists(num_v, constraints)

    # Track start time
    start_time = time.time()
    iteration = 0

    while True:
        iteration += 1
        all_solutions = []
        all_costs = []

        # Build ants
        for _ in range(num_ants):
            sol = build_ant_solution(
                num_v,
                not_feasible_before,
                crossing_vals,
                pheromone,
                pheromone_start,
                average_cost,
                alpha,
                beta
            )
            if sol is None:
                continue

            cost = consecutive_cost(sol, crossing_vals)
            all_solutions.append(sol)

            if not is_feasible(sol, constraints):
                print("THIS CANNOT HAPPEN #######################################")
                print(sol)
            all_costs.append(cost)

        # If no feasible solutions
        if not all_solutions:
            pheromone *= (1.0 - rho)
            print(f"Iter={iteration}, no feasible solutions.")
        else:
            # Local best
            idx_best = np.argmin(all_costs)
            local_best_sol = all_solutions[idx_best]
            local_best_cost = all_costs[idx_best]

            # Update global best
            if local_best_cost < best_cost:
                best_cost = local_best_cost
                best_perm = local_best_sol

            # Evaporate
            pheromone *= (1.0 - rho)
            pheromone_start *= (1.0 - rho)

            # Deposit
            for sol, cost in zip(all_solutions, all_costs):
                deposit = Q * best_cost / (1.0 + cost)
                pheromone_start[sol[0]] += deposit
                for i in range(len(sol) - 1):
                    a = sol[i]
                    b = sol[i+1]
                    pheromone[a, b] += deposit

            #print(f"Iter={iteration}, local best={local_best_cost:.4f}, global best={best_cost:.4f}")
            local_bests.append(local_best_cost)

        # Check time budget
        elapsed = time.time() - start_time
        if elapsed >= max_time_sec:
            print(f"\n[INFO] Time limit of {max_time_sec}s reached. Stopping.")
            break

    if __name__ == "__main__":
        if best_perm is None:
            print("[RESULT] No feasible solution found at all.")
        else:
            print("[RESULT] Best cost=", best_cost)
            print("Permutation:", " ".join([str(i) for i in best_perm]))
            feasible_ok = is_feasible(best_perm, constraints)
            print("Feasible check:", feasible_ok)
        return local_bests
    else:
        return best_cost



##############################################################################
#                           MAIN + PROFILE OUTPUT
##############################################################################

def print_profile_summary():
    """
    Prints a summary of times in profile_data: absolute and percentage.
    """
    total_time = sum(profile_data.values())
    print("\n=== PROFILE SUMMARY ===")
    if total_time < 1e-12:
        print("No profiled function calls or zero total time.")
        return

    # Sort from most time-consuming to least
    sorted_items = sorted(profile_data.items(), key=lambda x: -x[1])
    for fn_name, tsec in sorted_items:
        perc = (tsec / total_time) * 100.0
        print(f"{fn_name:35s} : {tsec:8.4f} s  ({perc:6.2f}%)")


if __name__ == "__main__":
    FILE_NAME = "tuning_instances/medium/inst_200_20_00001"  # Adjust if needed

    max_time = 10
    local_bests = ant_colony_opt(
        filename=FILE_NAME,
        num_ants=20,
        alpha=1.0,
        beta=1.0,
        rho=0.1,
        Q=0.5,
        max_time_sec=max_time,
        seed=0
    )

    # Print the profiling summary
    print_profile_summary()

    from math import log
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(local_bests, 'o', markersize=6/log(max_time))  # 'o' specifies a point graph
    plt.title("Point Graph of Floats")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)  # Add grid for better visualization

    # Save the plot as a PNG fileMWCCPGeneticAlgorithm
    plt.savefig("point_graph.png")
