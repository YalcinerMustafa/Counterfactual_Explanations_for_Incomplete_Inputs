from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import pandas as pd
import gurobipy
from network_encoders import Classifier
import numpy
from collections import namedtuple
from Config import Config
from util import CtxStatus, fetch_gurobi_env

Recourse = namedtuple('Recourse', ['cost', 'recourse', 'status'])


class SingleImputer:
    def __init__(self):
        self.imputer = IterativeImputer(sample_posterior=True)
        self.initialized = False

    def train(self, training_data_path):
        data_train_df = pd.read_csv(training_data_path)
        data_train_np = data_train_df.to_numpy()
        self.imputer.fit(data_train_np)

    def impute_single(self, missing_instance):
        return self.imputer.transform([missing_instance])[0]


class SingleImputationRecourse:
    def __init__(self):
        pass

    def single_impute_ctx_eval(self, imputation, path_to_classifier, target_class,
                               missing_dimensions):
        recourse, status = self.ctx_single_impute(imputation=imputation,
                                          path_to_classifier=path_to_classifier,
                                          target_class=target_class,
                                          immutable_dimensions=missing_dimensions)
        cost = numpy.sum(numpy.abs(numpy.asarray(recourse)))  # drop cost on missing value
        return Recourse(cost, recourse, status)

    def ctx_single_impute(self, imputation, path_to_classifier, target_class, immutable_dimensions):
        classifiers = []
        gurobi_model = gurobipy.Model('single_impute_model', env=fetch_gurobi_env())

        # Create names for variables for each dimension
        dimensions = len(imputation)
        new_vars = [f'diff_single_impute_vars{i}' for i in
                    range(dimensions)]  # fix this. fetch these variables like everything else.
        mul_result_vars = gurobi_model.addVars(new_vars, lb=-10000, ub=10000,
                                               vtype=gurobipy.GRB.CONTINUOUS)
        # We define the cost of the ctx. For that, we first need the action diff (i.e. ctx_i = x_i + diff_i)
        diff = [i for i in mul_result_vars.values()]

        classifier = Classifier(gurobi_model, naming_prefix=f'classifier_single_impute', input_vars=imputation,
                                target_class=target_class, path_to_classifier=path_to_classifier,
                                classifier_threshold=Config.classifier_threshold)
        classifier.encode_single_imput_ctx_classification_constraints(diff, Config.classifier_threshold)
        classifiers.append(classifier)

        diff_abs = classifiers[0].classifier_encoded.fetch_novel_variable_names(dimensions)
        for i in range(dimensions):
            gurobi_model.addConstr(diff_abs[i] == gurobipy.abs_(diff[i]), name=f'absolute_diff_dist{i}')
            gurobi_model.addConstr(diff_abs[i] <= 1, name=f'absolute_diff_is_sane_{i}')

        for dimension in immutable_dimensions:
            gurobi_model.addConstr(diff_abs[dimension] == 0,
                                   name=f'immutable_at_dimension{dimension}')

        gurobi_model.setObjective(gurobipy.quicksum(diff_abs), gurobipy.GRB.MINIMIZE)
        gurobi_model.optimize()

        # Check if optimization stopped due to timeout
        if gurobi_model.Status == gurobipy.GRB.TIME_LIMIT:
            if gurobi_model.SolCount > 0:
                return [val.X for val in diff], CtxStatus.Anytime
            else:
                return [], CtxStatus.Unsolved
        if gurobi_model.SolCount == 0:
            return [], CtxStatus.Unsolved
        return [val.X for val in diff], CtxStatus.Optimal
