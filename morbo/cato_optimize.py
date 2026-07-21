import sys
sys.path.append("/home/nafi/CATO/cato/hypermapper")
# import hypermapper
from hypermapper import optimizer
import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from helper import consts2
import warnings
import os
import time
import json
import datetime
import numpy as np
import argparse

from pprint import pprint

import pandas as pd
import pickle
from hypermapper.models.random_forest import RandomForest
from helper import consts
from helper import utils
from helper import prior_injection
from measure import measure_compute
from measure import measure_inference



# Filter out warnings
warnings.filterwarnings("ignore")

candidate_features = consts.candidate_features

# dimensionality reduction
mi = prior_injection.compute_mi_scores(candidate_features, pkt_depth="all")
candidate_features = [k for k,v in mi.items() if v > 0]

def gen_scenario_prior(candidate_features, max_pkt_depth, num_init, num_iter, include_priors, damping_factor, output_dir):
    """
    Generate and save a HyperMapper scenario with prior injection JSON.
    :param candidate_features: list of candidate features
    :param max_pkt_depth: int maximum packet depth to consider
    :param num_init: int number of initial samples in DoE phase
    :param num_iter: int number of optimization iterations
    :param include_priors: bool whether to include priors
    :param damping_factor: float damping factor over feature priors
    :return: str filename of saved scenario JSON
    """
    # each feature is a single input categorical parameter
    app_name = "cato"
    scenario = {}
    scenario["application_name"] = app_name

    output_file = os.path.join(
        output_dir,
        f"post_output_samples.csv",
    )
    scenario["output_data_file"] = output_file

    # minimize
    scenario["optimization_objectives"] = ["neg_f1_score", "compute_cost"]
    
    doe = {}
    doe["doe_type"] = "random sampling"
    doe["number_of_samples"] = num_init
    scenario["design_of_experiment"] = doe

    scenario["optimization_iterations"] = num_iter

    if include_priors:
        priors = prior_injection.compute_priors(candidate_features, pkt_depth="all", damping_factor=damping_factor)

    input_parameters = {}
    for candidate_feature in candidate_features:
        input_parameters[candidate_feature] = {
            "parameter_type": "categorical",
            "values": ["false", "true"],
        }
        if include_priors:
            input_parameters[candidate_feature]["prior"] = priors[candidate_feature]
    input_parameters["pkt_depth"] = {
        "parameter_type": "integer",
        "values": [1, max_pkt_depth],
    }
    if include_priors:
        input_parameters["pkt_depth"]["prior"] = "decay"
    scenario["input_parameters"] = input_parameters

    scenario_file = os.path.join(
        output_dir,
        f"scenario.json",
    )
    with open(scenario_file, 'w') as file:
        json.dump(scenario, file, indent=4)
    return scenario_file

def convert_point(x):
    """
    Converts point in HyperMapper search space to (feature_set, pkt_depth) tuple
    :param x: dict point in search space
    :return: tuple (list feature_set, int pkt_depth)
    """
    include_idx = np.where(
        [True if x[ft] == "true" else False for ft in candidate_features]
    )[0]
    feature_set = sorted([candidate_features[i] for i in include_idx])
    pkt_depth = int(x['pkt_depth'])
    return feature_set, pkt_depth

def objective_cato(x):
    """
    Optimize on measured perf and cost
    :param x: dict point in search space
    :return: dict HyperMapper optimization metrics
    """
    feature_set, pkt_depth = convert_point(x)
    print(utils.CYAN, feature_set, pkt_depth, utils.RESET)

    # replace with desired perf(x), cost(x)
    y_f1 = measure_inference.get_f1_score(feature_set, pkt_depth)  
    y_compute = measure_compute.get_compute_cost(feature_set, pkt_depth)

    # Cleanup: Delete the generated features_* directory
    feature_decimal = utils.feature_decimal(feature_set)
    dataset_dir = os.path.join(consts.dataset_dir, f"pkts_{pkt_depth}")
    model_dir = os.path.join(dataset_dir, f'features_{feature_decimal}')

    if os.path.exists(model_dir):
        try:
            import shutil
            shutil.rmtree(model_dir)
            print(utils.YELLOW + f"Deleted directory: {model_dir}" + utils.RESET)
        except OSError as e:
            # This should ideally not happen if shutil is used correctly
            print(utils.RED + f"Error deleting directory {model_dir}: {e}" + utils.RESET)

    optimization_metrics = {}
    optimization_metrics["neg_f1_score"] = -y_f1
    optimization_metrics["compute_cost"] = y_compute
    return optimization_metrics


def save_hm_native_surrogate(scenario_file, csv_file, output_dir):
    """
    Reads the scenario parameters, trains HyperMapper's native surrogate 
    model on the collected data, and saves it to the output directory.
    """
    # 1. Read parameters directly from the generated scenario
    with open(scenario_file, 'r') as f:
        scenario = json.load(f)
        
    objectives = scenario.get("optimization_objectives", [])
    input_params = scenario.get("input_parameters", {})
    
    # Extract HyperMapper's specific model parameters (defaulting to 10 if not set)
    hm_model_params = scenario.get("models", {})
    num_trees = hm_model_params.get("number_of_trees", 10)
    
    # 2. Load the optimization history
    df = pd.read_csv(csv_file)
    
    # 3. Format X data to match HyperMapper's internal expectations
    feature_cols = list(input_params.keys())
    X_df = df[feature_cols].copy()
    
    for col, props in input_params.items():
        if props.get("parameter_type") == "categorical":
            # HyperMapper internally maps 'true'/'false' to 1/0
            X_df[col] = X_df[col].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
            
    X = X_df.values
    
    # 4. Train a Native HM Model per objective
    surrogate_models = {}
    for obj in objectives:
        y = df[obj].values
        
        # Instantiate the model using the exact parameters from the scenario
        hm_model = RandomForest(
            parameters={"number_of_trees": num_trees} # Pass scenario configs here
        )
        hm_model.fit(X, y)
        surrogate_models[obj] = hm_model
        
    # 5. Serialize and save the dictionary of models
    pkl_path = os.path.join(output_dir, "hm_native_surrogate.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(surrogate_models, f)
        
    print(utils.GREEN + f"Native HM surrogate successfully saved to {pkl_path}" + utils.RESET)

def hm_run(candidate_features, max_pkt_depth, num_init, num_iter, include_priors, damping_factor, experiment_dir=""):
    """
    Run HyperMapper optimization and save results.
    :param candidate_features: list of candidate features
    :param max_pkt_depth: int maximum packet depth to consider
    :param num_init: int number of initial samples in DoE phase
    :param num_iter: int number of optimization iterations
    :param include_priors: bool whether to include priors
    :param damping_factor: float damping factor over feature priors
    :param experiment_dir: directory to put results
    :return: str scenario file path
    """
    candidate_decimal = utils.feature_decimal(candidate_features)
    output_dir = os.path.join(consts.results_dir, f"hmp_{candidate_decimal}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not include_priors:
        exp_dir = experiment_dir + "_np"
    else:
        exp_dir = experiment_dir
    hypermapper_dir = os.path.join(output_dir, exp_dir)
    if not os.path.exists(hypermapper_dir):
        os.mkdir(hypermapper_dir)

    # create new timestamped output directory
    dt = datetime.datetime.fromtimestamp(time.time())
    ts = dt.strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(hypermapper_dir, f"max{max_pkt_depth}_init{num_init}_iter{num_iter}_damp{damping_factor}_{ts}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # generate scenario and run HyperMapper
    scenario_file = gen_scenario_prior(candidate_features, max_pkt_depth, num_init, num_iter, include_priors, damping_factor, output_dir)
    start_ts = time.time()
    optimizer.optimize(scenario_file, objective_cato)
    end_ts = time.time()
    print(f"BO elapsed: {end_ts - start_ts}s")
    # Generate and save the native surrogate model
    output_csv = os.path.join(output_dir, "post_output_samples.csv")
    if os.path.exists(output_csv):
        try:
            save_hm_native_surrogate(scenario_file, output_csv, output_dir)
        except Exception as e:
            print(utils.RED + f"Failed to train native HM surrogate: {e}" + utils.RESET)
    else:
        print(utils.YELLOW + f"Output CSV not found. Skipping surrogate model generation." + utils.RESET)

    return scenario_file


def main(args):
    for i in range(args.num_trials):
        print(f"Trial {i+1}")
        for max_pkt_depth in args.max_pkt_depth.split(","):
            for num_init in args.num_init.split(","):
                for num_iter in args.num_iter.split(","):
                    for damping_factor in args.damping_factor.split(","):
                        print(max_pkt_depth, num_init, num_iter, args.priors, damping_factor)
                        hm_run(
                            candidate_features, 
                            int(max_pkt_depth), 
                            int(num_init), 
                            int(num_iter) - int(num_init), 
                            args.priors,
                            float(damping_factor), 
                            experiment_dir=args.experiment_dir
                        )
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CATO"
    )
    parser.add_argument("max_pkt_depth", type=str, help="Comma separated list of Maximum packet depth")
    parser.add_argument("num_init", type=str, help="Comma separated list of Number of initial samples to query")
    parser.add_argument("num_iter", type=str, help="Comma separated list of Number of BO iterations")
    parser.add_argument("damping_factor", type=str, help="Comma separated list of Dampen feature priors. 0 = no damping, 1 = no prior")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment output dir")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--priors", action="store_true", help="Include priors")
    parser.set_defaults(priors=True)
    main(parser.parse_args())
