import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
import matplotlib.pyplot as plt
from tqdm import tqdm
import more_itertools
from util import current_time_str
from more_itertools import collapse
import random
import time
from network_encoders import FlowImputerSingle
from network_encoders import TIMEOUT
from ConfigImputation import ConfigImputation
from util import get_commit_hash
from evaluate_flow_lof import LocalOutlierEstimator
import onnxruntime
from scipy.stats import norm
import math

def create_results_folder(base_dir="./experimental_results/imputation_compare"):
    """
    Create a timestamped folder for saving results.
    """
    timestamp = current_time_str()
    folder_path = os.path.join(base_dir, timestamp)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


results = {
    'Flow_l1': [],
    'Flow_l2': [],
    'Flow_linf': [],
    'Flow_runtime': [],
    'Flow_timeouts': [],
}


def flow_imputation_milp(missing_instance, path_to_flow):
    import gurobipy
    model = gurobipy.Model('imputation')# , env=fetch_gurobi_env()
    start_time = time.time_ns()
    imputed, _ , status = FlowImputerSingle().flow_imputation(missing_instance, path_to_flow, model)
    runtime = time.time_ns() - start_time
    return imputed, runtime, status


def compute_norms(errors):
    """
    Compute L1, L2, and Linf norms from errors.
    """
    if errors == []:
        return -1, -1, -1
    l1 = np.sum(np.abs(errors))
    l2 = np.sqrt(np.sum(np.square(errors)))
    linf = np.max(np.abs(errors))
    return np.round(l1,2), np.round(l2,2), np.round(linf,2)


def report_and_save_results(results, folder_path):
    """
    Save results to a CSV file in the folder.
    """
    csv_path = os.path.join(folder_path, "imputation_metrics.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)

    with open(f'{folder_path}/config_file.txt', "w") as file:
        file.write(ConfigImputation.to_string())
    print(f"Results saved to: {csv_path}")


def train_imputers(data_path_train):
    data_train_df = pd.read_csv(data_path_train)
    data_train_np = data_train_df.to_numpy()
    imp_mean = IterativeImputer(random_state=0)
    imp_mean.fit(data_train_np)

    knn_imputer = KNNImputer(n_neighbors=5)
    knn_imputer.fit(data_train_np)
    return imp_mean, knn_imputer


def compute_prob(val, mu=1, sigma=0.5):
    return norm.cdf((1 / sigma) * (math.log(val) - mu))

def plot_errors_vs_lof(lof_scores,flow_errors, results_folder, x_name, y_name):
    """
    Plot imputation errors (y-axis) against LOF scores (x-axis).
    """
    if not flow_errors or not lof_scores:
        print("No data to plot.")
        return

    # Ensure the data matches in indices
    if len(flow_errors) != len(lof_scores):
        raise ValueError("Length mismatch between flow_errors and lof_scores.")

    # Flatten lists if needed
    flow_errors_flat = np.array(list(collapse(flow_errors)))
    density_scores_flat = np.array(list(collapse(lof_scores)))
    # Assume flow_errors_flat and density_scores_flat are defined and are 1D numpy arrays
    # Define the bin edges for density: [0, 0.25), [0.25, 0.5), [0.5, 0.75), and [0.75, 1]
    bins = [0, 0.25, 0.5, 0.75, 1]
    boxplot_data = []
    labels = []

    # Create data for each bin
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        # For the last bin, include points equal to the upper bound
        if i == len(bins) - 2:
            mask = (density_scores_flat >= lower) & (density_scores_flat <= upper)
        else:
            mask = (density_scores_flat >= lower) & (density_scores_flat < upper)
        errors_in_bin = flow_errors_flat[mask]
        boxplot_data.append(errors_in_bin)
        labels.append(f"{lower}-{upper}")

    # Create the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(boxplot_data, labels=labels, patch_artist=True, showfliers=False)
    plt.xlabel("Density Level Set", fontsize=16)
    plt.ylabel("Imputation Error", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    # Save the plot
    plt.savefig(os.path.join(save_path, f"{dataset}-impute_boxplot.png"))
    plt.savefig(os.path.join(save_path, f"{dataset}-impute_boxplot.pgf"))
    print(f"Plot saved to: {save_path}")
    plt.show()
    return




def compute_density_est(instance):
    classifier_session = onnxruntime.InferenceSession(Path_to_flow_density_est)
    input_name = classifier_session.get_inputs()[0].name
    outputs = classifier_session.run(None, {input_name: np.asarray(instance).astype(np.float32)})
    return sum(abs(outputs[0]))

def main():
    # Create a timestamped folder for results
    results_folder = create_results_folder()

    data_df = pd.concat([pd.read_csv(PATH_TO_TEST)])
    data_np = data_df.to_numpy()

    random_dimensions = random.sample(range(0, len(data_np[0])), 1)
    all_dimension_combinations = [list(subset) for subset in more_itertools.powerset(random_dimensions)]
    all_dimension_combinations.remove([])

    #for dimensions_to_drop in tqdm(all_dimension_combinations, desc="Dimension combinations"):
    flow_errors = []
    lof_scores = []
    density_est_scores = []
    runtime_flow_sum = 0
    flow_timeouts = 0
    for i in range(1):
        for instance in tqdm(data_np[:min(ConfigImputation.experiment_count, len(data_np))]):  # Process first 10 test instances
            dimensions_to_drop = random.sample(range(len(instance)), 1)
            gt_values = [instance[dim] for dim in dimensions_to_drop]
            missing_instance = [np.nan if i in dimensions_to_drop else instance[i] for i in range(len(instance))]
            predictions_flow, runtime_flow, status = flow_imputation_milp(missing_instance, ConfigImputation.PATH_TO_FLOW_SMALL)
            imputed_instance = [predictions_flow[0] if i in dimensions_to_drop else instance[i] for i in range(len(instance))]
            density_est_scores.append(compute_prob(compute_density_est(imputed_instance)))
            if status == TIMEOUT:
                flow_timeouts = flow_timeouts + 1

            flow_error = np.abs(np.subtract(predictions_flow, gt_values))
            flow_errors.append(flow_error)
            lof_scores.append(local_outlier_est.predict([instance]))

        runtime_flow_sum = runtime_flow_sum + runtime_flow

    #plot_errors_vs_lof( lof_scores, flow_errors,results_folder, "lof", "err")
    plot_errors_vs_lof(density_est_scores, flow_errors,  results_folder, "Density level", "Imputation error")
    #plot_errors_vs_lof( density_est_scores, lof_scores,results_folder, "density_est", "lof")
    flow_errors = list(collapse(flow_errors))
    flow_l1, flow_l2, flow_linf = compute_norms(flow_errors)
    results["Flow_l1"].append(flow_l1)
    results["Flow_l2"].append(flow_l2)
    results["Flow_linf"].append(flow_linf)
    results["Flow_runtime"].append(round(runtime_flow_sum / 1000000000, 2))
    results["Flow_timeouts"].append(flow_timeouts)

    # Save all results to a CSV
    report_and_save_results(results, results_folder)

def empty_list():
    for key in results:
        results[key] = []


if __name__ == '__main__':

    ConfigImputation.commit_hash = get_commit_hash()
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",  # Set the TeX system
        "text.usetex": False,         # Disable using TeX for text rendering
        "pgf.rcfonts": False,         # Avoid using rc fonts for the PGF backend
        "pgf.preamble": "",
        "figure.figsize": (6, 4)
    })


    for dataset in ["diabetes"]:
        data_path = f'./experiment_sources/{dataset}'
        PATH_TO_TRAIN = f'{data_path}/train.csv'
        PATH_TO_VAL = f'{data_path}/val.csv'
        PATH_TO_TEST = f'{data_path}/test.csv'
        save_path = "./save_path/"
        local_outlier_est = LocalOutlierEstimator(PATH_TO_TRAIN, PATH_TO_VAL, PATH_TO_TEST)

        # breast cancer
        empty_list()
        ConfigImputation.PATH_TO_FLOW_SMALL = f'{data_path}/flow.onnx'
        Path_to_flow_density_est = f'{data_path}/backward_processed.onnx'
        ConfigImputation.DATA_PATH_TRAIN = PATH_TO_TRAIN
        ConfigImputation.DATA_PATH_TEST = PATH_TO_TEST
        main()