import numpy as np
import time
from testing import MWCCPGeneticAlgorithm
import os
import matplotlib.pyplot as plt
import csv


statistics = {}
instance_size = "medium_large"
instance_folder = "test_instances/" + instance_size
max_generations=2000
num_runs = 10
instances = [os.path.join(instance_folder, f) for f in os.listdir(instance_folder)]
output_file = f"statistics_{instance_size}.csv"

for instance in instances:
    file = instance
    ga = MWCCPGeneticAlgorithm(filename=file)
    mean_runtime = 0
    mean_objs = np.zeros(max_generations+1)
    best_objs = np.zeros(num_runs)
    print(instance)

    for i in range(num_runs):
        print("Run", i + 1)
        start = time.time()
        res = ga.run(population_size=36, mutation_rate=0.6535133792952, max_generations=max_generations, linear_scaling=False,
                     selection_pressure=2, frac_children=0.5866147196521, frac_elites=0.2157116543085)
        best_solution, best_fitness, best_objective, timeline = res
        end = time.time()
        best_objs[i] = best_objective
        mean_runtime += timeline[-1]["time"]
        for j in range(len(timeline)):
            mean_objs[j] += timeline[j]["objective"]

    mean_runtime /= num_runs
    mean_objs /= num_runs
    avg_times = np.linspace(0,mean_runtime,max_generations+1)
    statistics[file] = {"avg_times": avg_times,
                        "mean_objs": mean_objs,
                        "best_obj": np.mean(best_objs),
                        "stddev": np.std(best_objs)}



with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Instance", "Best Objective (Mean)", "Standard Deviation"])  # Header
    for instance, stats in statistics.items():
        writer.writerow([os.path.basename(instance), stats["best_obj"], stats["stddev"]])

print(f"Statistics saved to {output_file}")

runtime = 0
objs = np.zeros(max_generations+1)
for key in statistics.keys():
    runtime += statistics[key]["avg_times"][-1]
    objs += statistics[key]["mean_objs"]/statistics[key]["mean_objs"][0]

timepoints = np.linspace(0, runtime/len(instances), max_generations+1)
objs /= len(instances)

plt.plot(timepoints, objs)
plt.xlabel("time in s")
plt.ylabel("objective normalized to initial value")
plt.title("objective over time averaged over all " + instance_size +"  instances")
plt.savefig("obj_over_time_"+instance_size+".png", dpi=300, format='png')
plt.close()
# plt.show()

plt.plot([i for i in range(max_generations+1)], objs)
plt.xlabel("generation")
plt.ylabel("objective normalized to initial value")
plt.title("objective over generation averaged over all " + instance_size +"  instances")
plt.savefig("obj_over_gen_"+instance_size+".png", dpi=300, format='png')
# plt.show()