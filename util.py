import onnxruntime
import numpy as np

import scipy.stats
import math
from scipy.stats import norm

from datetime import datetime

import subprocess

from enum import Enum
import gurobipy
from Config import Config
from typing import List


class BaselineReport:
    def __init__(self):
        self.recourse: List = []  # List of recourse examples
        self.status: List = []  # List of CtxStatus objects (or another type)
        self.lof_score: List = []
        self.cost: List = []
        self.valid: List = []
        self.conf: List = []
        self.runtime: List = []


class DetailedReport:
    def __init__(self):# TODO specify dataset name and other hyperparams
        self.classifier_thold = -1
        self.plausibiliy = -1
        self.domain = "undefined"
        self.use_deep_classifier = False
        self.original_instances: List = []  # List of original instances
        self.missing_indices: List = []  # List of missing indices
        # Baseline attributes as instances of BaselineReport
        self.ours = BaselineReport()
        self.FIO = BaselineReport()
        self.armin = BaselineReport()
        self.single_impute = BaselineReport()
        self.dynamic_monolithic = BaselineReport()
        self.kd = BaselineReport()



gurobi_license = { # place your licence here
    "WLSACCESSID": "",
    "WLSSECRET": "",
    "LICENSEID": 123456789,
}

def fetch_gurobi_env(custom_timeout = None):
    env = gurobipy.Env(params=gurobi_license)
    env.setParam('OutputFlag', 0)
    if custom_timeout is None:
        env.setParam('TimeLimit', Config.timeout_milp)
    else:
        env.setParam('TimeLimit', custom_timeout)
    return env


class CtxStatus(Enum):
    Optimal = 0
    Anytime = 1
    Unsolved = 2

def obtain_classification(instance, path_to_classifier):
    classification, confidence = obtain_classification_and_conf(instance, path_to_classifier)
    return classification

def obtain_classification_and_conf(instance, path_to_classifier):
    classifier_session = onnxruntime.InferenceSession(path_to_classifier)
    input_name = classifier_session.get_inputs()[0].name
    outputs = classifier_session.run(None, {input_name: np.asarray(instance).astype(np.float32)})
    classification = outputs[0].argmax()
    dist_to_decision_boundary = outputs[0].max() - outputs[0].min()
    return classification, dist_to_decision_boundary


def quantile_log_normal(p, mu=1, sigma=0.5):
    return math.exp(mu + sigma * norm.ppf(p))


def quantile_log_normal_inverse(val, mu=1, sigma=0.5):
    return norm.cdf((math.log(val) - mu) / sigma)


def quantile_log_laplace(p, mu=1, b=0.5):
    return scipy.stats.laplace.ppf(p=p, loc=mu, scale=b)


def quantile_log_laplace_inverse(val, mu=1, sigma=0.5):
    return norm.cdf((math.log(val) - mu) / sigma)


def current_time_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_commit_hash():
    try:
        # Get the latest commit hash in short form (e.g., 'abc123')
        commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        return commit_hash
    except subprocess.CalledProcessError:
        # Handle the case where the Git command fails (e.g., not a Git repository)
        return "Failed_to_fetch_commit_hash"
