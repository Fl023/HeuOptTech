import time
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev
from solver import MWCCPSolver
import os


class MWCCPPlotter:
    def __init__(self, instance_file, seed=None):
        self.solver = MWCCPSolver(instance_file, seed=seed)

    def run_solver(self, algorithm="local_search", num_runs=10, step_function='best_improvement',
                   neighbors_function='swap_neighbors',segment_length=None, alpha=0.5,
                   max_grasp_iterations=100, max_ls_iterations=100):
        runtimes = []
        final_objectives = []
        iteration_counts = []
        timelines = []

        init_solution = self.solver.deterministic_construction_heuristic()

        neighbors_func_map = {
            'swap_neighbors': self.solver.swap_neighbors,
            'insert_neighbors': self.solver.insert_neighbors,
            'reverse_segment': lambda solution: self.solver.reverse_segment(solution, segment_length),
        }
        step_func_map = {
            'best_improvement': self.solver.best_improvement,
            'first_improvement': self.solver.first_improvement,
            'random': self.solver.random_step
        }
        neighbors_func = neighbors_func_map[neighbors_function]
        step_func = step_func_map[step_function]

        if algorithm == "local_search":
            solver = lambda: self.solver.local_search(init_solution, neighbors_func, step_func, max_ls_iterations)

        elif algorithm == "grasp":
            solver = lambda: self.solver.grasp(neighbors_func, step_func, alpha, max_grasp_iterations, max_ls_iterations)

        elif algorithm == "local_search_delta":
            neighbors_func_map = {
                'swap_neighbors': self.solver.swap_neighbors_delta,
                'insert_neighbors': self.solver.insert_neighbors_delta,
                'reverse_segment': lambda solution, step: self.solver.reverse_segment_delta(solution, step, segment_length),
            }
            neighbors_func = neighbors_func_map[neighbors_function]
            step_func = step_function
            solver = lambda: self.solver.local_search_delta(init_solution, neighbors_func, step_function, max_ls_iterations)

        else:
            raise ValueError("Algorithm not known.")

        for _ in range(num_runs):

            start_time = time.time()
            solutions = solver()
            end_time = time.time()

            # Record results
            runtimes.append(end_time - start_time)
            final_objectives.append(solutions[-1][2])
            iteration_counts.append(solutions[-1][0])
            timelines.append(solutions)

        return {
            "runtimes": runtimes,
            "final_objectives": final_objectives,
            "iteration_counts": iteration_counts,
            "timelines": timelines,
        }

    def plot_obj_over_time(self, results):
        timelines = results["timelines"]
        timeline = timelines[0]
        its = []
        objs = []
        runtimes = []
        for e in timeline:
            its.append(e[0])
            objs.append(e[2])
            runtimes.append(e[3]-timeline[0][3])
        plt.subplot(2, 1, 1)
        plt.plot(its, objs)
        plt.title("Objective Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Objective Value")

        plt.subplot(2, 1, 2)
        plt.plot(runtimes, objs)
        plt.title("Objective Over Time")
        plt.xlabel("runtime")
        plt.ylabel("Objective Value")
        plt.show()



if __name__ == "__main__":

    directory = "tuning_instances/small/"
    instance_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                      os.path.isfile(os.path.join(directory, f))]

    instance_file = "tuning_instances/small/inst_50_4_00010"
    plotter = MWCCPPlotter(instance_file, seed=42)

    num_runs = 10
    results = plotter.run_solver()
    plotter.plot_obj_over_time(results)

    # print(results["runtimes"])
    # print(results["final_objectives"])
    # print(results["iteration_counts"])