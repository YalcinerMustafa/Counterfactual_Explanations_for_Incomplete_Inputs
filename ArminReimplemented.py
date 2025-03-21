from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import pandas as pd
import gurobipy
from network_encoders import Classifier
import numpy
from util import obtain_classification
from collections import namedtuple
from Config import Config
from util import CtxStatus, fetch_gurobi_env

Recourse = namedtuple('Recourse', ['valid_ratio', 'cost', 'recourse', 'status'])


class ArminImputer:
    def __init__(self):
        self.imputer = IterativeImputer(sample_posterior=True)
        self.initialized = False

    def train(self, training_data_path):
        data_train_df = pd.read_csv(training_data_path)
        data_train_np = data_train_df.to_numpy()
        self.imputer.fit(data_train_np)

    def impute_single(self, missing_instance):
        return self.imputer.transform([missing_instance])[0]

    def impute_multi(self, missing_instance, n):
        imputations = []
        for i in range(n):
            imputations.append(self.imputer.transform([missing_instance])[0])
        return imputations

    def compute_imputations_sets(self, instance):  # see page 20 for def of N and P
        imputations = []  # contains sets of imputations.
        for i in range(Config.imputation_set_count):
            imputations.append(self.impute_multi(missing_instance=instance, n=Config.imputation_count_within_set))
        return imputations


class Armin:
    def __init__(self):
        pass

    def valid_ratio(self, classifications, target):
        return numpy.sum(numpy.array(classifications) == target) / len(classifications)

    def find_best_recourse(self, recourses: [Recourse]):
        recourses_prioritized = sorted(list(filter(lambda x: x.valid_ratio >= Config.validity_global, recourses)),
                                       key=lambda x: x.cost)
        if len(recourses_prioritized) > 0:
            # get the recourse action with the lowest cost among all high-validity actions.
            return recourses_prioritized[0]
        else:
            # get the recourse with the highest validity if all recourses are below the threshold.
            return sorted(recourses, key=lambda x: x.valid_ratio)[-1]

    def ctx_armin_meta(self, imputations_sets, path_to_classifier, target_class, classifier_threshold,
                       missing_dimension):
        recourses = []
        for imputations in imputations_sets:
            armin_sol, status = self.ctx_armin(imputations=imputations,
                                            path_to_classifier=path_to_classifier,
                                            target_class=target_class,
                                            immutable_dimensions=[missing_dimension],
                                            classifier_threshold=classifier_threshold)
            if not status == CtxStatus.Unsolved:
                recourses.append(armin_sol)
        # all_imputations_flat = numpy.array(imputations_sets).flatten().tolist()
        if recourses == []:
            return Recourse(-1,-1,[], CtxStatus.Unsolved)
        recourse_evals = []
        for recourse in recourses:
            recoursed_imputations = [imputation + recourse for imputation_set in imputations_sets for imputation in
                                     imputation_set]
            classifications = [obtain_classification(recoursed_imputation, path_to_classifier=path_to_classifier)
                               for recoursed_imputation in recoursed_imputations]
            validity = self.valid_ratio(classifications, target_class)
            cost = numpy.sum(numpy.abs(numpy.asarray(recourse)))  # drop cost on missing value
            recourse_evals.append(Recourse(validity, cost, recourse, CtxStatus.Optimal))
        return self.find_best_recourse(recourse_evals)

    def ctx_armin(self, imputations, path_to_classifier, target_class, classifier_threshold, immutable_dimensions):
        classifiers = []
        gurobi_model = gurobipy.Model('Armin_Wine_own',
                                      env=fetch_gurobi_env(round(Config.timeout_milp/Config.imputation_set_count)))

        # Create names for variables for each dimension
        dimensions = len(imputations[0])
        new_vars = [f'diff_armin_vars{i}' for i in
                    range(dimensions)]  # fix this. fetch these variables like everything else.
        mul_result_vars = gurobi_model.addVars(new_vars, lb=-10000, ub=10000,
                                               vtype=gurobipy.GRB.CONTINUOUS)
        # We define the cost of the ctx. For that, we first need the action diff (i.e. ctx_i = x_i + diff_i)
        diff = [i for i in mul_result_vars.values()]

        for i in range(len(imputations)):
            classifier = Classifier(gurobi_model, naming_prefix=f'classifier_armin{i}', input_vars=imputations[i],
                                    target_class=target_class, path_to_classifier=path_to_classifier,
                                    classifier_threshold=classifier_threshold)
            classifier.encode_target_classification_constraints(diff)
            classifiers.append(classifier)

        gurobi_model.addConstr(
            gurobipy.quicksum([classifier.output_is_target_binary_final for classifier in classifiers])
            >= Config.validity_within_milp * Config.imputation_count_within_set, name=f'armin_validity_pN')
        diff_abs = classifiers[0].classifier_encoded.fetch_novel_variable_names(dimensions)
        for i in range(dimensions):
            gurobi_model.addConstr(diff_abs[i] == gurobipy.abs_(diff[i]), name=f'absolute_diff_dist{i}')
            gurobi_model.addConstr(diff_abs[i] <= 1, name=f'absolute_diff_is_sane{i}')  # only for normed data

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
