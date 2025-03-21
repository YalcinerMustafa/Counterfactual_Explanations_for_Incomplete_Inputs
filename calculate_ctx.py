import os
import pickle
import time

import pandas as pd
import numpy as np
import gurobipy

import util
from network_encoders import FlowImputerSingle
from ArminReimplemented import Armin
from ArminReimplemented import ArminImputer
from SingleImputationRecourse import SingleImputer
from SingleImputationRecourse import SingleImputationRecourse
from dataset_manager import DatasetManager
from util import obtain_classification_and_conf, current_time_str
from flow_imputation_heuristic import DiverseFlowImputation
from Flow_ctx import calculate_recourse_action, calculate_all_in_one
from Config import Config
from local_outlier_estimator import LocalOutlierEstimator
import random
from util import get_commit_hash
from util import CtxStatus
from util import DetailedReport

import onnxruntime
from scipy.stats import norm
import math
from kd_tree import KDTreeWrap


def compute_imputations_flow(instance, PATH_TO_FLOW):
    only_imputed_values, full_imputed_instance, _ = FlowImputerSingle().flow_imputation(missing_instance=instance,
                                                                                     path_to_flow=PATH_TO_FLOW)
    return [full_imputed_instance]


def compute_prob(val, mu=1, sigma=0.5):
    return norm.cdf((1 / sigma) * (math.log(val) - mu))
def compute_density_est(instance):
    classifier_session = onnxruntime.InferenceSession(Config.PATH_TO_FLOW_DENSITY_EST)
    input_name = classifier_session.get_inputs()[0].name
    outputs = classifier_session.run(None, {input_name: np.asarray(instance[0]).astype(np.float32)})
    return compute_prob(sum(abs(outputs[0])))

def main():
    local_outlier_est = LocalOutlierEstimator()
    # data_df = pd.read_csv(DATA_PATH_TEST)
    # test_data_unlabeled = data_df.drop(labels=["Quality"], axis=1)

    # data_np = data_df.to_numpy()
    testset_manager = DatasetManager(path_to_dataset=Config.DATA_PATH_TEST)
    armin_imputer = ArminImputer()
    armin_imputer.train(Config.DATA_PATH_TRAIN)
    single_imputer = SingleImputer()
    single_imputer.train(Config.DATA_PATH_TRAIN)
    kd_tree = KDTreeWrap(Config.DATA_PATH_TRAIN)
    dataset_length = testset_manager.get_length()
    experiment_count = min(dataset_length-1, Config.experiment_count)
    random_integers = random.sample(population=range(0, experiment_count), k=experiment_count)

    detailed_report = DetailedReport()
    detailed_report.classifier_thold = Config.classifier_threshold
    detailed_report.plausibiliy = Config.optimize_plausibility
    detailed_report.domain = experiment_domain
    detailed_report.use_deep_classifier = use_deep_classifier
    for current_data_index in random_integers:
        preprocessed_instance = testset_manager.get_full_unlabeled_nan(current_data_index)
        classification, conf = obtain_classification_and_conf(preprocessed_instance.unlabled_instance, Config.PATH_TO_CLASSIFIER)
        missing_instance_changed_label = testset_manager.override_gt_label_with_classification(preprocessed_instance.missing_instance,
                                                                                                     classification)
        detailed_report.original_instances.append(preprocessed_instance.full_instance)
        detailed_report.missing_indices.append(preprocessed_instance.missing_dimension)
        target_class = 1 - classification  # Flip the class (for binary classification
        if Config.RUN_FLOW_IN_ONE:
            print(f'------------computing FIO ctx----------------------')
            current_time = time.time()
            recourse_action_all_in_one, status = calculate_all_in_one(missing_instance=missing_instance_changed_label.copy(),
                                                              target_class=target_class,
                                                              path_to_flow=Config.PATH_TO_FLOW,
                                                              path_to_classifier=Config.PATH_TO_CLASSIFIER,
                                                              target_index=testset_manager.index_of_label_column,
                                                              classifier_threshold=Config.classifier_threshold,
                                                              W_1=Config.FIO_opt_plausibility,
                                                              W_2=Config.FIO_opt_cost,
                                                              W_3=Config.FIO_opt_imputation,
                                                              missing_dimension=[preprocessed_instance.missing_dimension])

            runtime = (time.time() - current_time)
            detailed_report.FIO.recourse.append(recourse_action_all_in_one)
            detailed_report.FIO.status.append(status)
            detailed_report.FIO.runtime.append(runtime)
            if status == CtxStatus.Unsolved:
                detailed_report.FIO.cost.append(-1)
                detailed_report.FIO.valid.append(-1)
                detailed_report.FIO.lof_score.append(-1)
            else:
                cost = np.sum(np.abs(recourse_action_all_in_one))
                detailed_report.FIO.cost.append(cost)
                ctx_fio = preprocessed_instance.unlabled_instance + recourse_action_all_in_one
                ctx_fio_class, conf = obtain_classification_and_conf(ctx_fio, Config.PATH_TO_CLASSIFIER)
                detailed_report.FIO.conf.append(conf)
                print(f'ctx_ours_class is {ctx_fio_class}.')
                detailed_report.FIO.valid.append(ctx_fio_class == target_class)
                ctx_fio_labeled = ctx_fio.copy()
                # Test the following line
                ctx_fio_labeled = np.insert(ctx_fio_labeled, testset_manager.index_of_label_column, target_class)
                lof_score = local_outlier_est.predict(ctx_fio_labeled)
                detailed_report.FIO.lof_score.append(lof_score)


        if Config.RUN_KD:
            print(f'------------computing KD ctx ------------')
            current_time = time.time()
            imputation = single_imputer.impute_single(
                missing_instance_changed_label.copy())  # test changes
            imputation_unlabeled = testset_manager.remove_label(imputation)
            recourse = kd_tree.compute_recourse(imputation_unlabeled, target_class)
            runtime = (time.time() - current_time)
            detailed_report.kd.recourse.append(recourse)
            detailed_report.kd.status.append(CtxStatus.Optimal)
            detailed_report.kd.runtime.append(runtime)
            cost = np.sum(np.abs(recourse))
            detailed_report.kd.cost.append(cost)
            ctx_kd = preprocessed_instance.unlabled_instance + recourse
            ctx_kd_class, conf = obtain_classification_and_conf(ctx_kd, Config.PATH_TO_CLASSIFIER)
            detailed_report.kd.conf.append(conf)
            detailed_report.kd.valid.append(ctx_kd_class == target_class)
            ctx_kd_labeled = ctx_kd.copy()
            # Test the following line
            ctx_kd_labeled = np.insert(ctx_kd_labeled, testset_manager.index_of_label_column, target_class)
            lof_score = local_outlier_est.predict(ctx_kd_labeled)
            detailed_report.kd.lof_score.append(lof_score)


        if Config.RUN_DYNAMIC_MONOLITHIC:
            print(f'------------computing dynamic monolithic------------')
            current_time = time.time()
            imputed_flow_milp = compute_imputations_flow(instance=missing_instance_changed_label.copy(),
                                                         PATH_TO_FLOW=Config.PATH_TO_FLOW)
            density = compute_density_est(imputed_flow_milp)
            if density < 0.25:  # if we are in the 25% udl:
                Config.DM_opt_imputation = Config.udl_opt_impute_val
            elif density >= 0.25 and density <= 0.75:  # if we are in the 25% udl:
                Config.DM_opt_imputation = Config.mdl_opt_impute_val
            else:
                Config.DM_opt_imputation = Config.ldl_opt_impute_val

            recourse_action_all_in_one, status = calculate_all_in_one(missing_instance=missing_instance_changed_label.copy(),
                                                              target_class=target_class,
                                                              path_to_flow=Config.PATH_TO_FLOW,
                                                              path_to_classifier=Config.PATH_TO_CLASSIFIER,
                                                              target_index=testset_manager.index_of_label_column,
                                                              classifier_threshold=Config.classifier_threshold,
                                                              W_1=Config.DM_opt_plausibility,
                                                              W_2=Config.DM_opt_cost,
                                                              W_3=Config.DM_opt_imputation,
                                                              missing_dimension=[preprocessed_instance.missing_dimension])

            runtime = (time.time() - current_time)
            detailed_report.dynamic_monolithic.recourse.append(recourse_action_all_in_one)
            detailed_report.dynamic_monolithic.status.append(status)
            detailed_report.dynamic_monolithic.runtime.append(runtime)
            if status == CtxStatus.Unsolved:
                detailed_report.dynamic_monolithic.cost.append(-1)
                detailed_report.dynamic_monolithic.valid.append(-1)
                detailed_report.dynamic_monolithic.lof_score.append(-1)
            else:
                cost = np.sum(np.abs(recourse_action_all_in_one))
                detailed_report.dynamic_monolithic.cost.append(cost)
                ctx_fio = preprocessed_instance.unlabled_instance + recourse_action_all_in_one
                ctx_fio_class, conf = obtain_classification_and_conf(ctx_fio, Config.PATH_TO_CLASSIFIER)
                detailed_report.dynamic_monolithic.conf.append(conf)
                print(f'ctx_ours_class is {ctx_fio_class}.')
                detailed_report.dynamic_monolithic.valid.append(ctx_fio_class == target_class)
                ctx_fio_labeled = ctx_fio.copy()
                # Test the following line
                ctx_fio_labeled = np.insert(ctx_fio_labeled, testset_manager.index_of_label_column, target_class)
                lof_score = local_outlier_est.predict(ctx_fio_labeled)
                detailed_report.dynamic_monolithic.lof_score.append(lof_score)

        if Config.RUN_OURS:
            print(f'------------computing ours ctx------------')
            current_time = time.time()
            imputed_flow_milp = compute_imputations_flow(instance=missing_instance_changed_label.copy(),
                                                         PATH_TO_FLOW=Config.PATH_TO_FLOW)
            imputed_flow_milp_without_label = testset_manager.remove_label_array(imputed_flow_milp)

            recourse_action, status = calculate_recourse_action(imputed_instances=imputed_flow_milp_without_label,
                                                        target_class=target_class,
                                                        path_to_flow=Config.PATH_TO_FLOW,
                                                        path_to_classifier=Config.PATH_TO_CLASSIFIER,
                                                        target_index=testset_manager.index_of_label_column,
                                                        classifier_threshold=Config.classifier_threshold,
                                                        W_1=Config.optimize_plausibility,
                                                        W_2=Config.optimize_cost,
                                                        immutable_dimensions=[preprocessed_instance.missing_dimension])

            runtime = (time.time() - current_time)
            detailed_report.ours.recourse.append(recourse_action)
            detailed_report.ours.status.append(status)
            detailed_report.ours.runtime.append(runtime)
            if status == CtxStatus.Unsolved:
                detailed_report.ours.cost.append(-1)
                detailed_report.ours.valid.append(-1)
                detailed_report.ours.lof_score.append(-1)
            else:
                cost = np.sum(np.abs(recourse_action))
                ctx_ours = preprocessed_instance.unlabled_instance + recourse_action
                ctx_ours_labeled = ctx_ours.copy()
                ctx_ours_labeled = np.insert(ctx_ours_labeled, testset_manager.index_of_label_column, target_class)
                lof_score = local_outlier_est.predict(ctx_ours_labeled)
                ctx_ours_class, conf = obtain_classification_and_conf(ctx_ours, Config.PATH_TO_CLASSIFIER)
                detailed_report.ours.conf.append(conf)
                detailed_report.ours.cost.append(cost)
                detailed_report.ours.valid.append(ctx_ours_class == target_class)
                detailed_report.ours.lof_score.append(lof_score)
                print(f'ctx_ours_class is {ctx_ours_class}.')

        if Config.RUN_SINGLE_IMPUTE:
            print(f'------------computing single impute ctx ------------')
            current_time = time.time()
            imputation = single_imputer.impute_single(
                missing_instance_changed_label.copy())  # test changes
            imputation_unlabeled = testset_manager.remove_label(imputation)
            recourse_action= SingleImputationRecourse().single_impute_ctx_eval(
                                                     imputation=imputation_unlabeled,
                                                     path_to_classifier=Config.PATH_TO_CLASSIFIER,
                                                     target_class=target_class,
                                                     missing_dimensions=[preprocessed_instance.missing_dimension])
            runtime = (time.time() - current_time)
            detailed_report.single_impute.runtime.append(runtime)
            detailed_report.single_impute.recourse.append(recourse_action.recourse)
            detailed_report.single_impute.status.append(recourse_action.status)
            if recourse_action.status == CtxStatus.Unsolved:
                detailed_report.single_impute.cost.append(-1)
                detailed_report.single_impute.valid.append(-1)
                detailed_report.single_impute.lof_score.append(-1)
            else:
                ctx_single_impute = preprocessed_instance.unlabled_instance + recourse_action.recourse  #
                ctx_single_impute_labeled = ctx_single_impute.copy()
                ctx_single_impute_labeled = np.insert(ctx_single_impute_labeled, testset_manager.index_of_label_column,
                                                      target_class)
                lof_score = local_outlier_est.predict(ctx_single_impute_labeled)
                detailed_report.single_impute.lof_score.append(lof_score)
                detailed_report.single_impute.cost.append(recourse_action.cost)
                ctx_single_impute_class, conf = obtain_classification_and_conf(ctx_single_impute, Config.PATH_TO_CLASSIFIER)
                detailed_report.single_impute.conf.append(conf)

                detailed_report.single_impute.valid.append(ctx_single_impute_class == target_class)

        if Config.RUN_ARMIN:
            print(f'------------computing armin ctx------------')
            current_time = time.time()
            imputation_sets = armin_imputer.compute_imputations_sets(
                missing_instance_changed_label.copy())  # test changes
            imputation_sets_unlabeled = [testset_manager.remove_label_array(imputations) for imputations in
                                         imputation_sets]
            recourse_action = Armin().ctx_armin_meta(imputations_sets=imputation_sets_unlabeled,
                                                     path_to_classifier=Config.PATH_TO_CLASSIFIER,
                                                     target_class=target_class,
                                                     classifier_threshold=Config.classifier_threshold,
                                                     missing_dimension=preprocessed_instance.missing_dimension)
            runtime = (time.time() - current_time)
            detailed_report.armin.runtime.append(runtime)
            detailed_report.armin.recourse.append(recourse_action.recourse)
            detailed_report.armin.status.append(recourse_action.status)

            if recourse_action.status == CtxStatus.Unsolved:
                detailed_report.armin.cost.append(-1)
                detailed_report.armin.valid.append(-1)
                detailed_report.armin.lof_score.append(-1)
            else:
                ctx_armin = preprocessed_instance.unlabled_instance + recourse_action.recourse
                ctx_armin_labeled = ctx_armin.copy()
                ctx_armin_labeled = np.insert(ctx_armin_labeled, testset_manager.index_of_label_column, target_class)
                lof_score = local_outlier_est.predict(ctx_armin_labeled)
                detailed_report.armin.lof_score.append(lof_score)
                detailed_report.armin.cost.append(recourse_action.cost)
                print("done armin")
                ctx_armin_class, conf = obtain_classification_and_conf(ctx_armin, Config.PATH_TO_CLASSIFIER)
                detailed_report.armin.conf.append(conf)
                detailed_report.armin.valid.append(ctx_armin_class == target_class)
                print(f'ctx_armin_class is {ctx_armin_class}.')

        print(f'old classification was {classification}. New target was {target_class}')

    print(f'----------------------- saving experimental intermediate result -----------------------')

    detailed_report_path = (f'./experimental_results/{Config.detailed_report_folder_name}')
    if not os.path.exists(detailed_report_path):
        os.makedirs(detailed_report_path)
    # Save the object to a file
    file_path = f'{detailed_report_path}/detailed_deep_{use_deep_classifier}_clf_{round(Config.classifier_threshold,2)}_plaus_{Config.optimize_plausibility}-{experiment_domain}-{current_time_str()}.pkl'
    with open(file_path, "wb") as file:
        pickle.dump(detailed_report, file)
    # Save to CSV
    with open(f'{detailed_report_path}/config_file.txt', "w") as file:
        file.write(Config.to_string())
    print(f'--------------------------------------------------------------------------------------')


if __name__ == '__main__':
    Config.commit_hash = util.get_commit_hash()


    for iteration in range(1):
        for use_deep_classifier in [False, True]:
            for experiment_domain in ["power"]:
                for current_thold in np.arange(0, 10, 0.2):
                    Config.plausibility_constant_factor = current_thold
                    Config.set_experiment_folder(experiment_domain, use_deep_classifier=use_deep_classifier)
                    print(f'experiment for {experiment_domain}')
                    print(f'----starting with  clsf:{use_deep_classifier}, domain {experiment_domain}, clf_thold {Config.classifier_threshold}, plaus_factor {Config.optimize_plausibility}-----')
                    main()
                    print(f'----finished  clsf:{use_deep_classifier}, domain {experiment_domain}, clf_thold {Config.classifier_threshold}, plaus_factor {Config.optimize_plausibility}-----')
