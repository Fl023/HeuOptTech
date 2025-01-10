import os
import pickle
import numpy as np
from collections import defaultdict
import time
np.float_ = np.float64
np.complex_ = np.complex128

class MWCCPGeneticAlgorithm:
    def __init__(self, filename="tuning_instances/small/inst_50_4_00010", seed=0):
        self.profiling_times = defaultdict(float)  # To store profiling times
        self.num_u, self.num_v, self.constraints, self.edgesdic = self.read_instance(filename)
        self.nodes = np.arange(0, self.num_v, dtype=int)
        self.population_size = None
        self.population = None
        self.fitness_scores = None
        self.upper_tri_indices = np.triu_indices(self.num_v, k=1)
        self.best_solution = []
        self.best_fitness = 0
        self.tot = 0
        self.not_fine = 0
        self.tot_m = 0
        self.not_fine_m = 0

        # File paths for precomputed data
        precomputed_data_dir = "precomputed_data"
        os.makedirs(precomputed_data_dir, exist_ok=True)
        instance_name = os.path.basename(filename)
        self.precomputed_file = os.path.join(precomputed_data_dir, f"{instance_name}_data.pkl")

        # Load or calculate cross_vals and C_max
        self.crossing_vals = None
        precomputed_data = self.get_precomputed_data()
        self.crossing_vals = precomputed_data['cross_vals']
        self.C_max = precomputed_data['c_max']

        # print("seed:", seed)
        np.random.seed(seed)


    def profile(func):
        """Decorator to measure the runtime of functions."""

        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            self.profiling_times[func.__name__] += (end_time - start_time)
            return result

        return wrapper

    def read_instance(self, filename):
        # print("reading file ", filename, "...")
        with open(filename, 'r') as file:
            lines = file.readlines()

        first_line = lines[0].strip().split()
        num_u, num_v = int(first_line[0]), int(first_line[1])

        constraints_start = lines.index("#constraints\n") + 1
        edges_start = lines.index("#edges\n") + 1

        constraints = defaultdict(list)  # Directed Acyclic Graph
        for line in lines[constraints_start:edges_start - 1]:
            v, v_prime = map(int, line.strip().split())
            constraints[v-num_u-1].append(v_prime-num_u-1)

        # Read edges
        edgesdic = defaultdict(list)
        for line in lines[edges_start:]:
            u, v, weight = line.strip().split()
            edgesdic[int(v)-num_u-1].append((int(u)-1, float(weight)))

        edgesdicnumpy = defaultdict(lambda: np.array([[]]))
        for key, value in edgesdic.items():
            edgesdicnumpy[key] = np.array(value)

        return num_u, num_v, constraints, edgesdicnumpy

    def calculate_c_max(self):
        return sum(
            max(self.crossing_vals[i][j], self.crossing_vals[j][i])
            for i in range(self.num_v)
            for j in range(i + 1, self.num_v)
        )

    def calc_cross_vals(self):
        cross_vals = np.zeros((self.num_v, self.num_v))
        for i in self.nodes:
            edges_prime = self.edgesdic[i]
            if edges_prime.size == 0:
                continue
            u2, w2 = edges_prime[:, 0], edges_prime[:, 1]
            for j in self.nodes:
                if i == j:
                    continue
                edges = self.edgesdic[j]
                if edges.size == 0:
                    continue
                u1, w1 = edges[:, 0], edges[:, 1]
                total_crossings = np.sum((w1[:, None] + w2) * (u1[:, None] < u2))
                cross_vals[i][j] = total_crossings
        return cross_vals

    def calculate_objective(self, permutations):
        permuted_matrices = self.crossing_vals[permutations[:, :, None], permutations[:, None, :]]
        objectives = permuted_matrices[:, self.upper_tri_indices[0], self.upper_tri_indices[1]].sum(axis=1)
        return objectives

    def get_precomputed_data(self):
        """Load precomputed data or calculate and save it."""
        if os.path.exists(self.precomputed_file):
            # print(f"Loading precomputed data from {self.precomputed_file}")
            with open(self.precomputed_file, 'rb') as f:
                return pickle.load(f)
        else:
            # print(f"Precomputing data for instance.")
            cross_vals = self.calc_cross_vals()
            self.crossing_vals = cross_vals
            c_max = self.calculate_c_max()
            data = {'cross_vals': cross_vals, 'c_max': c_max}
            with open(self.precomputed_file, 'wb') as f:
                pickle.dump(data, f)
            return data

    def fitness_function(self, permutation):
        return self.C_max - self.calculate_objective(permutation)

    def linear_scaling(self, S):
        g_max = np.max(self.fitness_scores)
        g_min = np.min(self.fitness_scores)
        g_avg = np.mean(self.fitness_scores)

        if g_max != g_avg:
            a = (S * g_avg - g_avg) / (g_max - g_avg)
            b = g_avg - a * g_avg
        else:
            a = 1
            b = 0
        if g_min * a - b < 0:
            a = g_avg / (g_avg - g_min)
            b = a * g_avg - g_avg

        return np.maximum(a * self.fitness_scores - b, 0)

    def is_feasible(self, permutation):
        position = np.argsort(permutation)
        for v in self.constraints:
            for v_prime in self.constraints[v]:
                if position[v] >= position[v_prime]:
                    return False
        return True

    def initialize_population(self):
        self.population = np.array([np.random.permutation(self.nodes) for _ in range(self.population_size)])
        feasibility = np.array([self.is_feasible(i) for i in self.population])
        fine = np.sum(feasibility)
        not_fine = self.population_size - fine

        self.population[~feasibility] = np.array([self.repair(candidate) for candidate in self.population[~feasibility]])

        # print(f"{not_fine} / {self.population_size} candidates needed reparation")

    def evaluate_population(self):
        self.fitness_scores = self.fitness_function(self.population)

    def select_parents(self, num_parents, linear_scaling=True, selection_pressure=None):
        scores = self.fitness_scores
        if linear_scaling:
            scores = self.linear_scaling(selection_pressure)
        probabilities = scores / scores.sum()
        parent_indices = np.random.choice(self.population_size, size=num_parents, replace=True, p=probabilities)
        return self.population[parent_indices]

    def recombine(self, parents, num_children):
        offspring = []
        while len(offspring) < num_children:
            parent_indices = np.random.choice(len(parents), size=2, replace=False)
            parent1, parent2 = parents[parent_indices]
            child1, child2 = self.pmx(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)
        if len(offspring) > num_children:  # in case number of parents is odd
            offspring.pop()
        return offspring

    def pmx(self, parent1, parent2):
        point1, point2 = np.sort(np.random.randint(0, self.num_v, size=2))
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        temp1 = child1[point1:point2 + 1].copy()
        temp2 = child2[point1:point2 + 1].copy()
        child1[point1:point2 + 1] = temp2
        child2[point1:point2 + 1] = temp1
        return self.fix_duplicates(child1, parent1, point1, point2), self.fix_duplicates(child2, parent2, point1, point2)

    def fix_duplicates(self, child, donor, start, end):
        mapping = {child[i]: donor[i] for i in range(start, end + 1)}
        for i in list(range(start)) + list(range(end + 1, len(child))):
            while child[i] in mapping:
                child[i] = mapping[child[i]]
        if not self.is_feasible(child):
            child = self.repair(child)
            self.not_fine += 1
        self.tot += 1
        return child

    def repair(self, chromosome):
        position = np.argsort(chromosome)
        repaired = True
        while repaired:
            repaired = False
            for v in self.constraints:
                for v_prime in self.constraints[v]:
                    if position[v] >= position[v_prime]:
                        chromosome = np.delete(chromosome, position[v])
                        chromosome = np.insert(chromosome, position[v_prime], v)
                        position = np.argsort(chromosome)
                        repaired = True
        return chromosome

    def mutate(self, chromosomes, mutation_rate):
        mutated_chroms = []
        for chromosome in chromosomes:
            if np.random.rand(1)[0] < mutation_rate:
                i, j = np.random.randint(0, self.num_v, size=2)
                chromosome[[i, j]] = chromosome[[j, i]]
                if not self.is_feasible(chromosome):
                    chromosome = self.repair(chromosome)
                    self.not_fine_m += 1
                self.tot_m += 1
                mutated_chroms.append(chromosome)
            else:
                mutated_chroms.append(chromosome)
        return mutated_chroms

    def replace_population(self, children, elite_count=1):
        elite_indices = sorted(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i], reverse=True)[
                        :elite_count]
        elites = self.population[elite_indices]
        elites_fitness = self.fitness_scores[elite_indices]

        permuted_children_indices = np.random.permutation(len(children))[
                                    :min(len(children), self.population_size - elite_count)]
        selected_children = children[permuted_children_indices]
        children_fitness = self.fitness_function(selected_children)

        remaining_slots = self.population_size - elite_count - len(selected_children)
        permuted_old_indices = np.random.permutation(len(self.population))[:remaining_slots]
        selected_old = self.population[permuted_old_indices]
        selected_old_fitness = self.fitness_scores[permuted_old_indices]

        new_population = np.concatenate((selected_children, selected_old, elites))
        self.population = new_population
        self.fitness_scores = np.concatenate((children_fitness, selected_old_fitness, elites_fitness))

    def run(self, population_size=10, mutation_rate=1.0, max_generations=100, linear_scaling=True,
            selection_pressure=2.0, frac_children=1.0, frac_elites=0.1):
        self.population_size = population_size
        self.population = np.empty((self.population_size, self.num_v), dtype=int)
        self.fitness_scores = np.zeros(self.population_size)
        num_children = int(frac_children*population_size)
        num_parents = max(2,num_children)
        num_elites = int(frac_elites*population_size)

        self.best_fitness = 0
        self.tot = 0
        self.not_fine = 0
        self.tot_m = 0
        self.not_fine_m = 0

        self.initialize_population()
        self.evaluate_population()
        best_idx = np.argmin(self.fitness_scores)
        if self.best_fitness < self.fitness_scores[best_idx]:
            self.best_solution = self.population[best_idx]
            self.best_fitness = self.fitness_scores[best_idx]
        print(f"Generation {0}: Best Fitness = {self.best_fitness}")
        for generation in range(max_generations):
            parents = self.select_parents(num_parents, linear_scaling=linear_scaling,
                                          selection_pressure=selection_pressure)
            children = self.recombine(parents, num_children)
            children = self.mutate(children, mutation_rate)
            self.replace_population(np.array(children), elite_count=num_elites)
            # self.evaluate_population()

            best_idx = np.argmin(self.fitness_scores)
            if self.best_fitness < self.fitness_scores[best_idx]:
                self.best_solution = self.population[best_idx]
                self.best_fitness = self.fitness_scores[best_idx]
            print(f"Generation {generation+1}: Best Fitness = {self.best_fitness}, Objective = {self.C_max - self.best_fitness}")


        # print("For PMX", self.not_fine, "/", self.tot, "chromosomes needed reparation.")
        # print("For mutation", self.not_fine_m, "/", self.tot_m, "chromosomes needed reparation.")
        # print("\nProfiling times (in seconds):")
        # for func, runtime in self.profiling_times.items():
        #     print(f"{func}: {runtime:.6f} s")
        return self.best_solution + 1 + self.num_u, self.best_fitness, self.C_max - self.best_fitness


if __name__ == "__main__":
    start = time.time()
    ga = MWCCPGeneticAlgorithm(filename="tuning_instances/small/inst_50_4_00010")
    end = time.time()
    print("Initialization runtime:", end - start, "s")
    num_runs = 1

    for i in range(num_runs):
        print("\nRun", i+1)
        start = time.time()
        res = ga.run(population_size=13, mutation_rate=1.0, max_generations=1000, linear_scaling=True,
                     selection_pressure=2, frac_children=0.46477, frac_elites=0.095553)
        best_solution, best_fitness, best_objective = res
        end = time.time()
        print()
        print("Best Solution:", best_solution)
        print("Best Fitness:", best_fitness)
        print("Best Objective:", best_objective)