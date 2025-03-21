import pickle

import util
from util import DetailedReport
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def plot_baseline_comparison(report: DetailedReport, name):
    # Define baselines and criteria
    baselines = {
        #"Ours": report.ours,
        #"FIO": report.FIO,
        "Ours": report.dynamic_monolithic,
        "ARMIN": report.armin,
        "NNCE": report.kd,
        "SI": report.single_impute,
    }

    criteria = ["lof_score", "cost", "runtime"]
    num_criteria = len(criteria)

    # Initialize a figure with subplots
    fig, axes = plt.subplots(1, num_criteria, figsize=(5 * num_criteria, 5), sharey=False)
    for ax in axes.flat:  # Iterate over all subplot axes
        ax.tick_params(axis='y', labelsize=16)  # Increases y-axis font size
        ax.tick_params(axis='x', labelsize=16)  # Also increases x-axis font size if needed

    #fig.suptitle(name, fontsize=16)

    for i, criterion in enumerate(criteria):
        ax = axes[i] if num_criteria > 1 else axes  # Handle single criterion case

        # Extract data for each baseline for the current criterion
        data = {
            baseline: [value for value in getattr(baseline_data, criterion) if value != -1]
            for baseline, baseline_data in baselines.items()
        }
        # Prepare boxplot data (for overview across baselines)
        box_data = [values for values in data.values()]
        if False: #criterion == "conf"
            data_valid = {
                baseline: [value for value in getattr(baseline_data, "valid") if value != -1]
                for baseline, baseline_data in baselines.items()
            }
            baseline_names = [
                f"{baseline} ({sum(values) / len(values):.2f})"
                if len(values) > 0 else f"{baseline} (0.00)"
                for baseline, values in data_valid.items()
            ]
        else:
            baseline_names = list([key.replace("_","-") for key in data.keys()])

        criterion_to_full_name = {
            "lof_score": "Plausibility (LOF-Score)",
            "cost": "Cost (L1)",
            "runtime": "Runtime (Sec. per instance)",
            "conf": "Distance to Decision Boundary"
        }
        criterion_to_short_name = {
            "lof_score": "LOF",
            "cost": "Cost",
            "runtime": "Runtime",
            "conf": "Dist to Boundary"
        }

        # Boxplot
        ax.boxplot(box_data, labels=baseline_names, patch_artist=True, showfliers=False)
        #ax.set_xticklabels(baseline_names, fontsize=16)  # Adjust font size
        #plt.yticks(fontsize=14)  # Adjust the font size as needed
        ax.set_title(criterion_to_full_name[criterion], fontsize=16)
        #ax.set_ylabel(criterion_to_full_name[criterion])
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f'{SAVE_TO_PATH}/{name}.png')
    #rc_params_old = plt.rcParams.copy()
    plt.savefig(f'{SAVE_TO_PATH}/{name}.pgf')
    #plt.rcParams = rc_params_old
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.show()



# Example usage:
if __name__ == "__main__":
    # Create dummy data for demonstration
    from random import uniform
    import numpy as np
    import sys

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",  # Set the TeX system
        "text.usetex": False,         # Disable using TeX for text rendering
        "pgf.rcfonts": False,         # Avoid using rc fonts for the PGF backend
        "pgf.preamble": "",
        "figure.figsize": (6, 4)
    })
    print(sys.executable)
    print(np.__version__)
    folder_path = "./benchmark_results"
    SAVE_TO_PATH = "./benchmark_results/plots"  # save the pngs to the same folder that also contains the data.
    # Load the object from the file
    reports = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pkl"):
                if True: #plot_dataset in file.split("/")[-1] and use_deep in file.split("/")[-1]:
                    file_path = os.path.join(root, file) #f'{folder_path}/detailed_report.pkl'
                    with open(file_path, "rb") as file:
                        loaded_report: DetailedReport = pickle.load(file)
                        reports.append(loaded_report)

    reports.sort(key=lambda x: (x.domain,x.use_deep_classifier))

    READ_ARMIN = True
    READ_OURS = False
    READ_FIO = False
    READ_SI = True
    READ_DYN_MONOLITHIC = True
    READ_KD = True

    for report_deep_classifier in [True, False]:
        domain_reports = defaultdict(list)
        armin_valids = defaultdict(list)
        FIO_valids = defaultdict(list)
        ours_valids = defaultdict(list)
        SI_valids = defaultdict(list)
        dyn_mon_valids = defaultdict(list)
        kd_valids = defaultdict(list)
        for report in reports:
            if report.use_deep_classifier == report_deep_classifier:
                domain_reports[report.domain].append(report)
        for domain in domain_reports.keys():
            for report in domain_reports[domain]:
                for i in range(len(domain_reports[domain])):
                    if READ_ARMIN:
                        armin_valids[domain].append(sum(domain_reports[domain][i].armin.valid)/len(domain_reports[domain][i].armin.valid))
                    if READ_FIO:
                        FIO_valids[domain].append(sum(domain_reports[domain][i].FIO.valid)/len(domain_reports[domain][i].FIO.valid))
                    if READ_OURS:
                        ours_valids[domain].append(sum(domain_reports[domain][i].ours.valid)/len(domain_reports[domain][i].ours.valid))
                    if READ_SI:
                        SI_valids[domain].append(sum(domain_reports[domain][i].single_impute.valid)/len(domain_reports[domain][i].single_impute.valid))
                    if READ_DYN_MONOLITHIC:
                        dyn_mon_valids[domain].append(sum(domain_reports[domain][i].dynamic_monolithic.valid)/len(domain_reports[domain][i].dynamic_monolithic.valid))
                    if READ_KD:
                        kd_valids[domain].append(sum(domain_reports[domain][i].kd.valid)/len(domain_reports[domain][i].kd.valid))

        print(f'..........CLASSIFIER_DEEP? {report_deep_classifier}.............................')
        domains = ["diabetes", "concrete", "wine", "power"]
        methods = ["armin", "single_impute", "dyn_mono", "kd"]
        performance_data = {
            "armin": armin_valids,
            "single_impute": SI_valids,
            "dyn_mono": dyn_mon_valids,
            "kd": kd_valids
        }

        output_lines = []

        for method in methods:
            line = f"{method.replace('_', ' ').title()} & "
            for domain in domains:
                mean = round(np.mean(performance_data[method][domain]), 2)
                std = round(np.std(performance_data[method][domain]), 2)
                value_str = f"${mean}\pm{std}$"
                line += value_str + " & "
            line = line.rstrip(" & ") + " \\\\"
            output_lines.append(line)

        # Print the formatted lines
        for line in output_lines:
            print(line)


    meta_report = defaultdict(lambda: defaultdict(DetailedReport))

    for report in reports:
        domain = report.domain
        use_deep = report.use_deep_classifier
        meta_report[domain][use_deep].use_deep_classifier = use_deep
        meta_report[domain][use_deep].classifier_thold = report.classifier_thold
        meta_report[domain][use_deep].plausibiliy = report.plausibiliy
        meta_report[domain][use_deep].domain = report.domain
        meta_report[domain][use_deep].original_instances = report.original_instances
        meta_report[domain][use_deep].missing_indices = report.missing_indices

        if READ_OURS:
            meta_report[domain][use_deep].ours.recourse.extend(report.ours.recourse)
            meta_report[domain][use_deep].ours.status.extend(report.ours.status)
            meta_report[domain][use_deep].ours.lof_score.extend(report.ours.lof_score)
            meta_report[domain][use_deep].ours.cost.extend(report.ours.cost)
            meta_report[domain][use_deep].ours.valid.extend(report.ours.valid)
            meta_report[domain][use_deep].ours.conf.extend(report.ours.conf)
            meta_report[domain][use_deep].ours.runtime.extend(report.ours.runtime)
            meta_report[domain][use_deep].FIO.recourse.extend(report.FIO.recourse)
            meta_report[domain][use_deep].FIO.status.extend(report.FIO.status)
            meta_report[domain][use_deep].FIO.lof_score.extend(report.FIO.lof_score)
            meta_report[domain][use_deep].FIO.cost.extend(report.FIO.cost)
            meta_report[domain][use_deep].FIO.valid.extend(report.FIO.valid)
            meta_report[domain][use_deep].FIO.conf.extend(report.FIO.conf)
            meta_report[domain][use_deep].FIO.runtime.extend(report.FIO.runtime)

        if READ_ARMIN:
            meta_report[domain][use_deep].armin.recourse.extend(report.armin.recourse)
            meta_report[domain][use_deep].armin.status.extend(report.armin.status)
            meta_report[domain][use_deep].armin.lof_score.extend(report.armin.lof_score)
            meta_report[domain][use_deep].armin.cost.extend(report.armin.cost)
            meta_report[domain][use_deep].armin.valid.extend(report.armin.valid)
            meta_report[domain][use_deep].armin.conf.extend(report.armin.conf)
            meta_report[domain][use_deep].armin.runtime.extend(report.armin.runtime)

        if READ_SI:
            meta_report[domain][use_deep].single_impute.recourse.extend(report.single_impute.recourse)
            meta_report[domain][use_deep].single_impute.status.extend(report.single_impute.status)
            meta_report[domain][use_deep].single_impute.lof_score.extend(report.single_impute.lof_score)
            meta_report[domain][use_deep].single_impute.cost.extend(report.single_impute.cost)
            meta_report[domain][use_deep].single_impute.valid.extend(report.single_impute.valid)
            meta_report[domain][use_deep].single_impute.conf.extend(report.single_impute.conf)
            meta_report[domain][use_deep].single_impute.runtime.extend(report.single_impute.runtime)

        if READ_DYN_MONOLITHIC:
            meta_report[domain][use_deep].dynamic_monolithic.recourse.extend(report.dynamic_monolithic.recourse)
            meta_report[domain][use_deep].dynamic_monolithic.status.extend(report.dynamic_monolithic.status)
            meta_report[domain][use_deep].dynamic_monolithic.lof_score.extend(report.dynamic_monolithic.lof_score)
            meta_report[domain][use_deep].dynamic_monolithic.cost.extend(report.dynamic_monolithic.cost)
            meta_report[domain][use_deep].dynamic_monolithic.valid.extend(report.dynamic_monolithic.valid)
            meta_report[domain][use_deep].dynamic_monolithic.conf.extend(report.dynamic_monolithic.conf)
            meta_report[domain][use_deep].dynamic_monolithic.runtime.extend(report.dynamic_monolithic.runtime)

        if READ_KD:
            meta_report[domain][use_deep].kd.recourse.extend(report.kd.recourse)
            meta_report[domain][use_deep].kd.status.extend(report.kd.status)
            meta_report[domain][use_deep].kd.lof_score.extend(report.kd.lof_score)
            meta_report[domain][use_deep].kd.cost.extend(report.kd.cost)
            meta_report[domain][use_deep].kd.valid.extend(report.kd.valid)
            meta_report[domain][use_deep].kd.conf.extend(report.kd.conf)
            meta_report[domain][use_deep].kd.runtime.extend(report.kd.runtime)

    if not os.path.exists(SAVE_TO_PATH):
        os.makedirs(SAVE_TO_PATH)



    for reports_all_domains in [val for val in meta_report.values()]:
        for reports_nn_lin in reports_all_domains.values():
            report = reports_nn_lin
            if util.CtxStatus.Unsolved in loaded_report.FIO.status:
                print(f'unsolved status in {file.name.split("/")[-1]}')
            if util.CtxStatus.Anytime in loaded_report.FIO.status:
                print(f'anytime status in {file.name.split("/")[-1]}')
            plot_baseline_comparison(report, f'dataset {report.domain} - classifier: {"neural network" if report.use_deep_classifier else "linear"}')
