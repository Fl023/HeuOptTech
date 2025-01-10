import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition, Float, Integer
from genetic_algorithm import MWCCPGeneticAlgorithm

cs = ConfigurationSpace(seed=1234)

c = Categorical("linear_scaling", items=[True, False], default=True)
f = Float("selection_pressure", bounds=(1.2, 2.0), default=2.0)
f2 = Float("frac_children", bounds=(0.1, 1.0), default=0.5)
f3 = Float("mutation_rate", bounds=(0.0, 1.0), default=1.0)
f4 = Float("frac_elites", bounds=(0.0, 0.3), default=0.1)
i1 = Integer("population_size", bounds=(5, 500), log=True, default=20)


# A condition where `f` is only active if `c` is equal to True when sampled
cond = EqualsCondition(f, c, True)

# Add them explicitly to the configuration space
cs.add([c, f, f2, f3, f4, i1])
cs.add(cond)

print(cs)


def train(config, instance=None, seed=None,) -> float:
    print(config)
    population_size = config["population_size"]
    frac_children = config["frac_children"]
    linear_scaling = config["linear_scaling"]
    selection_pressure = None
    if linear_scaling:
        selection_pressure = config["selection_pressure"]
    mutation_rate = config["mutation_rate"]
    frac_elites = config["frac_elites"]
    num_runs = 10
    objectives = np.empty(num_runs)
    ga = MWCCPGeneticAlgorithm(filename=instance, seed=seed)
    for i in range(num_runs):
        _, _, best_objective = ga.run(
            population_size=population_size,
            mutation_rate=mutation_rate,
            max_generations=1000,
            linear_scaling=linear_scaling,
            selection_pressure=selection_pressure,
            frac_children=frac_children,
            frac_elites=frac_elites
        )
        objectives[i] = best_objective

    return 0 + np.mean(objectives)

from smac import Scenario

instances = ["tuning_instances/small/inst_50_4_00001",
             "tuning_instances/small/inst_50_4_00002",
             "tuning_instances/small/inst_50_4_00003",
             "tuning_instances/small/inst_50_4_00004",
             "tuning_instances/small/inst_50_4_00005",
             "tuning_instances/small/inst_50_4_00006",
             "tuning_instances/small/inst_50_4_00007",
             "tuning_instances/small/inst_50_4_00008",
             "tuning_instances/small/inst_50_4_00009",
             "tuning_instances/small/inst_50_4_00010"]

indices = [[float(i)] for i in range(len(instances))]

instance_features = dict(zip(instances, indices))

scenario = Scenario(
    configspace=cs,
    walltime_limit=60*15,  # Limit to 15 minutes
    n_trials=500,  # Evaluated max 500 trials
    deterministic=True,  # It's not, but we average over n runs in train()
    instances=instances,
    instance_features=instance_features,
)

from smac import AlgorithmConfigurationFacade as ACFacade
smac = ACFacade(scenario=scenario, target_function=train)
incumbent = smac.optimize()
print("Best Configuration:", incumbent)