from scipy.spatial import KDTree
import pandas as pd
import numpy as np
from util import obtain_classification_and_conf
from Config import Config

class KDTreeWrap():
    def __init__(self, path_to_data):
        data_train_df = pd.read_csv(path_to_data).to_numpy()
        self.X = data_train_df[:, :-1]  # All rows, all columns except the last
        self.Y = data_train_df[:, -1].astype(int)  # All rows, the last column
        self.pred = np.array([obtain_classification_and_conf(instance, Config.PATH_TO_CLASSIFIER)[0] for instance in self.X])

        #self.ts.fit(self.X, self.Y, classes=2)
        classes = 2
        self.kdtrees = [None] * classes
        for c in range(classes):
            X_fit = self.X[np.where(self.pred == c)[0]]
            self.kdtrees[c] = KDTree(X_fit, leafsize=40,copy_data=True)
        print("asdf")

    def compute_recourse(self, point, target_label):
        dists, ids = self.kdtrees[target_label].query([point], k=1, p=1)
        ctx = self.kdtrees[target_label].data[ids[0]]
        recourse_action = ctx - point
        return recourse_action



