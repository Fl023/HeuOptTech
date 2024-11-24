import time
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
from solver import MWCCPSolver
import os
import multiprocessing
import concurrent.futures

class MWCCPPlotter:
    def __init__(self, instance_file, seed=None):
        self.solver = MWCCPSolver(instance_file, seed=seed)

    def run_solver(self, algorithm="VND_delta", num_runs=10, step_function='best_improvement',
                   neighbors_function='swap_neighbors', segment_length=None, alpha=0.5,
                   max_grasp_iterations=100, max_ls_iterations=100, max_vnd_iterations=1000, max_vnd_swaps=100,
                   max_sa_iterations=70,
                   sa_it=1000, sa_cr=0.99, sa_mt=1):
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
            solver = lambda: self.solver.grasp(neighbors_func, step_func, alpha, max_grasp_iterations,
                                               max_ls_iterations)

        elif algorithm == "local_search_delta":
            neighbors_func_map = {
                'swap_neighbors': self.solver.swap_neighbors_delta,
                'insert_neighbors': self.solver.insert_neighbors_delta,
                'reverse_segment': lambda solution, step: self.solver.reverse_segment_delta(solution, step,
                                                                                            segment_length),
            }
            neighbors_func = neighbors_func_map[neighbors_function]
            step_func = step_function
            solver = lambda: self.solver.local_search_delta(init_solution, neighbors_func, step_function,
                                                            max_ls_iterations)

        elif algorithm == "grasp_delta":
            neighbors_func_map = {
                'swap_neighbors': self.solver.swap_neighbors_delta,
                'insert_neighbors': self.solver.insert_neighbors_delta,
                'reverse_segment': lambda solution, step: self.solver.reverse_segment_delta(solution, step,
                                                                                            segment_length),
            }
            neighbors_func = neighbors_func_map[neighbors_function]
            step_func = step_function
            solver = lambda: self.solver.grasp_delta(neighbors_func, step_func, alpha, max_grasp_iterations,
                                                     max_ls_iterations)

        elif algorithm == "VND":
            solver = lambda: self.solver.solve_VND(init_solution=init_solution, step_function_string=step_function,
                                                   max_neigborhood_swaps=max_vnd_swaps,
                                                   max_iterations_per_neighborhood=max_vnd_iterations)

        elif algorithm == "VND_delta":
            solver = lambda: self.solver.solve_VND(init_solution=init_solution, step_function_string=step_function, 
                                                   neighborhood_functions=["swap_neighbors", "reverse_segment"],
                                                   # care, need strings
                                                   max_neigborhood_swaps=max_vnd_swaps, segment_length=5,
                                                   max_iterations_per_neighborhood=max_vnd_iterations, use_delta=True)

        # solve_simulated_annealing(self, init_solution_func="topological_sort", neighbors_func='swap_neighbors',
        #                            initial_temperature=1000, cooling_rate=0.90, stopping_temperature=0.01,
        #                            max_iterations=1000, segment_length=5, use_delta=False, init_solution = None):

        elif algorithm == "SA":
            solver = lambda: self.solver.solve_simulated_annealing(init_solution=init_solution,
                                                                   neighbors_func=neighbors_function,
                                                                   # care, need strings
                                                                   max_iterations=max_sa_iterations, segment_length=5,
                                                                   use_delta=False,
                                                                   cooling_rate=sa_cr, initial_temperature=sa_it,
                                                                   stopping_temperature=sa_mt)

        elif algorithm == "SA_delta":
            solver = lambda: self.solver.solve_simulated_annealing(init_solution=init_solution,
                                                                   neighbors_func=neighbors_function,
                                                                   # care, need strings
                                                                   max_iterations=max_sa_iterations, segment_length=5,
                                                                   use_delta=True,
                                                                   cooling_rate=sa_cr, initial_temperature=sa_it,
                                                                   stopping_temperature=sa_mt)

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

    def plot_obj_over_time(self, results_arr, filename="graph.png", suptitel="",
                           titels=["Plot 1", "Plot 2", "Plot 3", "Plot 4"]):
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D']
        colors = ['blue', 'green', 'red', 'purple']

        plt.figure(figsize=(10, 6))  # Set figure size

        for i, results in enumerate(results_arr):
            timelines = results["timelines"]
            timeline = timelines[0]
            its = []
            objs = []
            runtimes = []
            for e in timeline:
                its.append(e[0])
                objs.append(e[2])
                runtimes.append(e[3] - timeline[0][3])

            plt.plot(runtimes, objs, label=titels[i],
                     linestyle=line_styles[i % len(line_styles)],
                     color=colors[i % len(colors)],
                     linewidth=2)

            plt.title(suptitel)
            plt.xlabel("Runtime /s")
            plt.ylabel("Objective Value")

            plt.grid(True)

            plt.legend()

            plt.tight_layout()

            plt.savefig(filename, format='png', dpi=300)

    def plot_localsearch(self, num_runs=10):
        neighborhoods = ['swap_neighbors', 'insert_neighbors', 'reverse_segment']
        steps = ['first_improvement', 'best_improvement', 'random']
        runtimes = []
        runtimes_std = []
        final_objectives = []
        final_objectives_std = []
        iterations = []
        objectives = []
        runtimes_iterations = []
        labels = []
        for neighborhood in neighborhoods:
            for step in steps:
                print(neighborhood, step)
                res = self.run_solver(algorithm="local_search_delta", num_runs=num_runs,
                                         step_function=step, neighbors_function=neighborhood,
                                         segment_length=4, max_ls_iterations=100)

                runtimes.append(np.mean(res["runtimes"]))
                runtimes_std.append(np.std(res["runtimes"]))
                final_objectives.append(np.mean(res["final_objectives"]))
                final_objectives_std.append(np.std(res["final_objectives"]))

                timelines = res["timelines"]
                timeline = timelines[0]
                its = []
                objs = []
                runtimes_its = []
                for e in timeline:
                    its.append(e[0])
                    objs.append(e[2])
                    runtimes_its.append(e[3] - timeline[0][3])

                iterations.append(its)
                objectives.append(objs)
                runtimes_iterations.append(runtimes_its)
                labels.append(neighborhood + " with " + step + " step")

        # Bar Plot 1: Runtimes
        plt.figure(figsize=(12, 6))
        x = np.arange(len(runtimes))
        plt.bar(x, runtimes, yerr=runtimes_std, capsize=5, alpha=0.75, label='Runtimes')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel("Runtime in s")
        plt.title("Mean Runtimes of Local Search")
        plt.tight_layout()
        plt.savefig("runtimes_localsearch.png", format='png', dpi=300)

        # Bar Plot 2: Final Objectives
        plt.figure(figsize=(12, 6))
        plt.bar(x, final_objectives, yerr=final_objectives_std, capsize=5, alpha=0.75, label='Final Objectives')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel("Final Objective")
        plt.title("Mean Final Objective of Local Search")
        plt.tight_layout()
        plt.savefig("objectives_localsearch.png", format='png', dpi=300)

        # Line Plots: Objective over Iterations and Runtime
        plt.figure(figsize=(12, 6))
        for i in range(len(iterations)):
            plt.plot(iterations[i], objectives[i], label=labels[i])
        plt.xlabel("Iterations")
        plt.ylabel("Objective")
        plt.title("Objective over Iterations for Local Search")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("objectives_over_iterations_localsearch.png", format='png', dpi=300)

        plt.figure(figsize=(12, 6))
        for i in range(len(iterations)):
            plt.plot(runtimes_iterations[i], objectives[i], label=labels[i])
        plt.xlabel("Runtime in s")
        plt.ylabel("Objective")
        plt.title("Objective over Runtime for Local Search")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("objectives_over_time_localsearch.png", format='png', dpi=300)

    def plot_localsearch_delta_vs_not(self, num_runs=10):
        neighborhoods = ['swap_neighbors', 'insert_neighbors', 'reverse_segment']
        steps = ['first_improvement', 'best_improvement', 'random']
        runtimes = []
        runtimes_std = []
        labels = []
        runtimes2 = []
        runtimes_std2 = []

        for neighborhood in neighborhoods:
            for step in steps:
                print(neighborhood, step)
                res = self.run_solver(algorithm="local_search_delta", num_runs=num_runs,
                                         step_function=step, neighbors_function=neighborhood,
                                         segment_length=4, max_ls_iterations=100)
                res2 = self.run_solver(algorithm="local_search", num_runs=num_runs,
                                      step_function=step, neighbors_function=neighborhood,
                                      segment_length=4, max_ls_iterations=100)
                runtimes.append(np.mean(res["runtimes"]))
                runtimes_std.append(np.std(res["runtimes"]))
                runtimes2.append(np.mean(res2["runtimes"]))
                runtimes_std2.append(np.std(res2["runtimes"]))
                labels.append(neighborhood + " with " + step + " step")

        # Bar Plot 1: Runtimes
        plt.figure(figsize=(12, 6))
        x = np.arange(len(runtimes))
        plt.bar(x, runtimes, yerr=runtimes_std, capsize=5, alpha=0.75, label='Runtimes')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel("Runtime in s")
        plt.title("Mean Runtimes of Local Search with Delta Evaluation")
        plt.tight_layout()
        plt.savefig("runtimes_localsearch_with.png", format='png', dpi=300)

        plt.figure(figsize=(12, 6))
        x = np.arange(len(runtimes2))
        plt.bar(x, runtimes2, yerr=runtimes_std2, capsize=5, alpha=0.75, label='Runtimes')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel("Runtime in s")
        plt.title("Mean Runtimes of Local Search without Delta Evaluation")
        plt.tight_layout()
        plt.savefig("runtimes_localsearch_without.png", format='png', dpi=300)

    def plot_all(self, num_runs=10):
        print("local search")
        local_search = self.run_solver(algorithm="local_search_delta", num_runs=num_runs,
                                       step_function="best_improvement", neighbors_function="reverse_segment",
                                       segment_length=4, max_ls_iterations=1000)
        print("GRASP")
        grasp = self.run_solver(algorithm="grasp_delta", num_runs=num_runs,
                                step_function="best_improvement", neighbors_function="reverse_segment",
                                segment_length=4, max_ls_iterations=1000, max_grasp_iterations=3, alpha=0.5)
        print("VND")
        vnd = self.run_solver(algorithm="VND_delta", num_runs=num_runs)
        print("SA")
        sa = self.run_solver(algorithm="SA_delta", num_runs=num_runs)

        results = [local_search, grasp, vnd, sa]
        labels = ["local search", "GRASP", "VND", "SA"]
        runtimes = []
        runtimes_std = []
        final_objectives = []
        final_objectives_std = []
        iterations = []
        objectives = []
        runtimes_iterations = []

        for res in results:
            runtimes.append(np.mean(res["runtimes"]))
            runtimes_std.append(np.std(res["runtimes"]))
            final_objectives.append(np.mean(res["final_objectives"]))
            final_objectives_std.append(np.std(res["final_objectives"]))

            timelines = res["timelines"]
            timeline = timelines[0]
            its = []
            objs = []
            runtimes_its = []
            for e in timeline:
                its.append(e[0])
                objs.append(e[2])
                runtimes_its.append(e[3] - timeline[0][3])

            iterations.append(its)
            objectives.append(objs)
            runtimes_iterations.append(runtimes_its)

        # Bar Plot 1: Runtimes
        plt.figure(figsize=(12, 6))
        x = np.arange(len(runtimes))
        plt.bar(x, runtimes, yerr=runtimes_std, capsize=5, alpha=0.75, label='Runtimes')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel("Runtime in s")
        plt.title("Mean Runtimes of Different Algorithms")
        plt.tight_layout()
        plt.savefig("runtimes_all.png", format='png', dpi=300)

        # Bar Plot 2: Final Objectives
        plt.figure(figsize=(12, 6))
        plt.bar(x, final_objectives, yerr=final_objectives_std, capsize=5, alpha=0.75, label='Final Objectives')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel("Final Objective")
        plt.title("Mean Final Objective of Different Algorithms")
        plt.tight_layout()
        plt.savefig("objectives_all.png", format='png', dpi=300)

        # Line Plots: Objective over Runtime

        plt.figure(figsize=(12, 6))
        for i in range(len(iterations)):
            plt.plot(runtimes_iterations[i], objectives[i], label=labels[i])
        plt.xlabel("Runtime in s")
        plt.ylabel("Objective")
        plt.title("Objective over Runtime for Different Algorithms")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig("objectives_over_time_all.png", format='png', dpi=300)

sa_it_arr = [1e2, 1e4, 1e6, 1e8]
sa_cr_arr = [0.7, 0.9, 0.95, 0.99]
max_sa_iterations_arr = [5, 20, 50, 100]

def run_solver_with_max_sa_iterations(i):
    return plotter.run_solver(
        algorithm="SA_delta",
        max_vnd_iterations=1000,
        max_vnd_swaps=40,
        num_runs=num_runs,
        sa_it=sa_it,
        sa_mt=sa_mt,
        sa_cr=sa_cr,
        max_sa_iterations=max_sa_iterations_arr[i]
    )

def run_solver_with_sa_it(i):
    return plotter.run_solver(
        algorithm="SA_delta",
        max_vnd_iterations=1000,
        max_vnd_swaps=40,
        num_runs=num_runs,
        sa_it=sa_it_arr[i],
        sa_mt=sa_mt,
        sa_cr=sa_cr,
        max_sa_iterations=max_sa_iterations
    )

def run_solver_with_sa_cr(i):
    return plotter.run_solver(
        algorithm="SA_delta",
        max_vnd_iterations=1000,
        max_vnd_swaps=40,
        num_runs=num_runs,
        sa_it=sa_it,
        sa_mt=sa_mt,
        sa_cr=sa_cr_arr[i],
        max_sa_iterations=max_sa_iterations
    )

if __name__ == "__main__":

    instance_file = "test_instances/medium/inst_200_20_00001"
    plotter = MWCCPPlotter(instance_file, seed=42)
    num_runs = 10
    plotter.plot_all(num_runs=num_runs)
    #with concurrent.futures.ProcessPoolExecutor() as executor:
        #future1 = executor.submit(plotter.plot_localsearch, num_runs=num_runs)
        #future2 = executor.submit(plotter.plot_localsearch_delta_vs_not, num_runs=num_runs)
        #future3 = executor.submit(plotter.plot_all, num_runs=num_runs)
        
        #concurrent.futures.wait([future1, future2, future3])
    exit()


    directory = "tuning_instances/medium/"
    instance_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                      os.path.isfile(os.path.join(directory, f))]

    instance_file = "tuning_instances/medium/inst_200_20_00001"
    plotter = MWCCPPlotter(instance_file, seed=42)

    num_runs = 10

    results_arr = list()
    sa_it = 10000
    sa_mt = 1e-7
    sa_cr = 0.95
    max_sa_iterations = 40

    with multiprocessing.Pool(processes=4) as pool:
        results_arr = pool.map(run_solver_with_sa_it, range(4))
        plotter.plot_obj_over_time(
            results_arr,
            filename="it.png",
            titels=[str(i) for i in sa_it_arr],
            suptitel="Objective function vs Execution Time for different Initial Temps"
        )

    with multiprocessing.Pool(processes=4) as pool:
        results_arr = pool.map(run_solver_with_sa_cr, range(4))
        plotter.plot_obj_over_time(
            results_arr,
            filename="cr.png",
            titels=[str(i) for i in sa_cr_arr],
            suptitel="Objective function vs Execution Time for different Cooling Rates"
        )

    with multiprocessing.Pool(processes=4) as pool:
        results_arr = pool.map(run_solver_with_max_sa_iterations, range(4))
        plotter.plot_obj_over_time(
            results_arr,
            filename="iterations.png",
            titels=[str(i) for i in max_sa_iterations_arr],
            suptitel="Objective function vs Execution Time for different Iteration Counts"
        )
