from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from Config import Config


class LocalOutlierEstimator(BaseEstimator):
    def __init__(self):
        train = pd.read_csv(Config.DATA_PATH_TRAIN)
        val = pd.read_csv(Config.DATA_PATH_VAL)
        test = pd.read_csv(Config.DATA_PATH_TEST)
        # The normalizing flow paper for ctx seems to evaluate the lof score on training data.
        # https://github.com/ofurman/counterfactuals/blob/main/counterfactuals/metrics/metrics.py#L19
        # https://github.com/ofurman/counterfactuals/blob/main/notebooks/ppcef.ipynb
        # That might actually make sense as the primary goal is not evaluating the quality of the flow model
        # For capturing the data distribution

        self.dataset = pd.concat([train, test, val], ignore_index=True)
        self.dataset.columns = range(self.dataset.shape[1])
        self.local_outlier_estimator = LocalOutlierFactor(n_neighbors=Config.lof_neighbor_count, novelty=True)
        self.local_outlier_estimator.fit(self.dataset)

    def predict(self, ctx):  # 1 inlier | -1 outlier
        return self.local_outlier_estimator.decision_function(ctx.reshape(1,-1))[0]
