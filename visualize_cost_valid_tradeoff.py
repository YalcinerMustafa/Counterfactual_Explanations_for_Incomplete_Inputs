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
    folder_path = "./tradeoff_cost_val_results/cost-valid-tradeoff-fixed"
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
    filtered_reports = [report for report in reports if report.use_deep_classifier and report.domain == "power"]

    dyn_mon_valids = defaultdict(float)
    for i in range(len(filtered_reports)):
        dyn_mon_valids[(sum(filtered_reports[i].dynamic_monolithic.cost)/len(filtered_reports[i].dynamic_monolithic.cost))] = (sum(filtered_reports[i].dynamic_monolithic.valid)/len(filtered_reports[i].dynamic_monolithic.valid))

    # Sort data by classifier threshold
    sorted_thresholds = sorted(dyn_mon_valids.keys())
    sorted_values = [dyn_mon_valids[thold] for thold in sorted_thresholds]



    # Create plot
    plt.figure(figsize=(6, 4))
    plt.scatter(sorted_thresholds, sorted_values, marker='o', color='b')
    # Customize plot
    plt.xlabel('Cost', fontsize=16)
    plt.ylabel('Valid ratio', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('')
    plt.grid(True)
    plt.tight_layout()

    # Save as PGF
    #plt.savefig('dyn_mon_valids.pgf')

    plt.savefig(f'{SAVE_TO_PATH}/dyn_mon_valids.png')
    plt.savefig(f'{SAVE_TO_PATH}/dyn_mon_valids.pgf')

    plt.show()