import pickle

import util
from util import DetailedReport
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict



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
    })
    print(sys.executable)
    print(np.__version__)
    folder_path = "./tradeoff_plaus_val/plausibility-validity-power-big-20"
    SAVE_TO_PATH = folder_path  # save the pngs to the same folder that also contains the data.
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
    print(len(reports))
    domain = "power"
    filtered_reports = [report for report in reports if report.use_deep_classifier and report.domain == domain]

    dyn_mon_valids = defaultdict(float)
    dyn_mon_lofs = defaultdict(float)
    for i in range(len(filtered_reports)):
        dyn_mon_valids[filtered_reports[i].plausibiliy] = (sum(filtered_reports[i].dynamic_monolithic.valid)/len(filtered_reports[i].dynamic_monolithic.valid))
        dyn_mon_lofs[filtered_reports[i].plausibiliy] = (sum(filtered_reports[i].dynamic_monolithic.lof_score)/len(filtered_reports[i].dynamic_monolithic.lof_score))

    # Sort data by classifier threshold
    sorted_thresholds = sorted(dyn_mon_valids.keys())
    sorted_values_valids = [dyn_mon_valids[thold] for thold in sorted_thresholds]
    sorted_values_lofs = [dyn_mon_lofs[thold] for thold in sorted_thresholds]



    # Create plot
    plt.figure(figsize=(6, 4))
    plt.scatter(sorted_thresholds, sorted_values_valids, marker='o', color='b')

    # Customize plot
    plt.xlabel('Plausibility', fontsize=16)
    plt.ylabel('Valid ratio', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('')
    plt.grid(True)
    plt.tight_layout()

    # Save as PGF
    #plt.savefig('dyn_mon_valids.pgf')

    plt.savefig(f'{SAVE_TO_PATH}/dyn_mon_valids-nn.png')
    plt.savefig(f'{SAVE_TO_PATH}/dyn_mon_valids-nn.pgf')

    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(sorted_thresholds, sorted_values_lofs, marker='o', linestyle='-', color='g', label='Plausibility vs lof score')

    # Customize plot
    plt.xlabel('Plausibility')
    plt.ylabel('LOF Avg')
    plt.title('plausibility vs LOF Avg')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save as PGF
    #plt.savefig('dyn_mon_valids.pgf')

    plt.savefig(f'{SAVE_TO_PATH}/dyn_mon_valids-nn-2.png')
    plt.savefig(f'{SAVE_TO_PATH}/dyn_mon_valids-nn-2.pgf')
    plt.show()