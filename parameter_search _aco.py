#!/usr/bin/env python3

import numpy as np
from smac import Scenario
from smac import AlgorithmConfigurationFacade as ACFacade
from ConfigSpace import ConfigurationSpace, Float, Integer

##############################################################################
#                 Import your ACO function from aco.py
##############################################################################
# ant_colony_opt now ONLY returns a single float = best cost
from aco import ant_colony_opt

##############################################################################
#                Define your hyperparameter configuration space
##############################################################################
cs = ConfigurationSpace(seed=1234)

hp_num_ants = Integer("num_ants", (10, 200), default=20, log=True)
hp_alpha    = Float("alpha", (0.4, 3.0), default=1.0, log=False)
hp_beta     = Float("beta", (0.4, 3.0), default=1.0, log=False)
hp_rho      = Float("rho", (0.02, 0.3), default=0.1, log=False)
hp_Q        = Float("Q", (0.1, 10.0),  default=1.0, log=False)

cs.add_hyperparameters([hp_num_ants, hp_alpha, hp_beta, hp_rho, hp_Q])

##############################################################################
#                   Helper: Run SMAC for a given instance set
##############################################################################
def run_smac_for_instances(instances, n_trials, walltime, max_time_sec):
    """
    Runs SMAC for a given list of instances.
    
    Arguments:
    -----------
    instances    : list of instance file paths
    n_trials     : max # of SMAC trials
    walltime     : SMAC's own overall time limit (seconds)
    max_time_sec : time budget for ACO (passed directly to ant_colony_opt)

    Returns:
    --------
    incumbent : best found configuration
    """
    # 1) Build scenario
    indices = [[float(i)] for i in range(len(instances))]
    instance_features = dict(zip(instances, indices))

    scenario = Scenario(
        configspace=cs,
        n_trials=n_trials,
        walltime_limit=walltime,
        deterministic=False,
        instances=instances,
        instance_features=instance_features,
    )

    # 2) Define the local train function
    def train(config, instance=None, seed=None) -> float:
        """
        SMAC calls this for each hyperparameter config + instance.
        We do multiple runs to reduce noise, then return the mean cost.
        """
        #print(config)
        #print(instance)
        num_runs = 1
        costs = []
        for i in range(num_runs):
            # ant_colony_opt now returns a single float = best cost
            best_cost = ant_colony_opt(
                filename=instance,
                num_ants=config["num_ants"],
                alpha=config["alpha"],
                beta=config["beta"],
                rho=config["rho"],
                Q=config["Q"],
                max_time_sec=max_time_sec,  # <<< time budget for ACO
                seed=(seed + i if seed is not None else None),
            )
            costs.append(best_cost)
        #print("returning cost:" , np.mean(costs))
        return float(np.mean(costs))

    # 3) Build and run SMAC
    smac = ACFacade(scenario=scenario, target_function=train)
    incumbent = smac.optimize()
    return incumbent

##############################################################################
#                         MAIN: Three calls for S/M/L
##############################################################################
if __name__ == "__main__":
    # Example instance sets
    instances_small = [
        "tuning_instances/small/inst_50_4_00001",
        "tuning_instances/small/inst_50_4_00010",
        # ...
    ]
    instances_medium = [
        "tuning_instances/medium/inst_200_20_00001",
        "tuning_instances/medium/inst_200_20_00010",
        # ...
    ]
    instances_medium_large = [
        "tuning_instances/medium_large/inst_500_40_00002.txt",
        "tuning_instances/medium_large/inst_500_40_00023.txt",
        # ...
    ]
    instances_large = [
        "tuning_instances/large/inst_1000_60_00001",
        "tuning_instances/large/inst_1000_60_00010",
        # ...
    ]

    time_small  = 12   # seconds for small
    time_medium = 40   # seconds for medium
    time_medium_large = 100   # seconds for medium
    time_large  = 220   # seconds for large

    # SMAC settings
    n_trials_small  = 30000000
    n_trials_medium = 30000000
    n_trials_large  = 30000000

    # SMAC walltime (seconds) for each run
    walltime_small  = 600
    walltime_medium = 2000
    walltime_medium_large = 5000
    walltime_large  = 11000

    print("\n=== Running SMAC on MEDIUM_LARGE INSTANCES ===")
    best_conf_medium_large = run_smac_for_instances(
        instances_medium_large,
        n_trials=n_trials_medium,
        walltime=walltime_medium_large,
        max_time_sec=time_medium_large
    )

    print("=== Running SMAC on SMALL INSTANCES ===")
    best_conf_small = run_smac_for_instances(
        instances_small,
        n_trials=n_trials_small,
        walltime=walltime_small,
        max_time_sec=time_small
    )
    
    print("\n=== Running SMAC on MEDIUM INSTANCES ===")
    best_conf_medium = run_smac_for_instances(
        instances_medium,
        n_trials=n_trials_medium,
        walltime=walltime_medium,
        max_time_sec=time_medium
    )
    
    print("\n=== Running SMAC on LARGE INSTANCES ===")
    best_conf_large = run_smac_for_instances(
        instances_large,
        n_trials=n_trials_large,
        walltime=walltime_large,
        max_time_sec=time_large
    )
    print("Best config (SMALL):", best_conf_small)
    print("Best config (MEDIUM):", best_conf_medium)
    print("Best config (MEDIUM_LARGE):", best_conf_medium_large)
    print("Best config (LARGE):", best_conf_large)
