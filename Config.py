

class Config:
    # ATTENTION: The missing dimension must come before the target label in the dataset.
    # Because we don't keep track of the index of the missing dimension after removing the target label!

    commit_hash = "Not_fetched"  # The commit hash of the current codebase
    detailed_report_folder_name = "plausibility-validity-power-big-20"
    experiment_count = 100000

    RUN_ARMIN = False
    RUN_OURS = False
    RUN_FLOW_IN_ONE = False
    RUN_SINGLE_IMPUTE = False
    RUN_DYNAMIC_MONOLITHIC = True
    RUN_KD = False

    # 60 Seconds for any MILP instance (same as for armin)
    timeout_milp = 60  # 100 seconds
    # ---------Config ours OG -------------
    plausibility_constant_factor = 0
    # Config ours:
    optimize_plausibility = 0.5
    optimize_cost = 0.5  # 1 - W_1
    # ctx_ours_imputations = 1  # if you want to enable this, change the code in the ctx method to heuristic instead of milp

    # ---------Config ours ALL-IN-ONE -------------
    FIO_opt_plausibility = 0.33
    FIO_opt_cost = 0.33
    FIO_opt_imputation = 0.33

    # ---------- Config dynamic monolithic-----
    DM_opt_plausibility = None
    DM_opt_cost = None
    DM_opt_imputation = None

    udl_opt_impute_val = 4
    mdl_opt_impute_val = 3
    ldl_opt_impute_val = 2

    # ---------Config used by both ours OG and ours ALL-IN-ONE and the baseline single-imputation------------
    classifier_threshold = 1.0
    classifier_thold_factor = 4.3

    # ------------------ Config Armin
    imputation_count_within_set = 10  # Indicates the number of imputations encoded into the MILP (heuristic implementation)
    imputation_set_count = 10  # Number of subcalls to the MILP = number of sets that contain a set of imputations (yes, a set of sets)
    validity_within_milp = 0.75  # Indicates the encoded valid ratio within the MILP
    validity_global = 0.75  # Indicates the targeted (not guaranteed) valid ratio for the obtained global solution

    # LOF score
    lof_neighbor_count = 20  # default is 20

    __use_deep_classifier = False

    __experiment_path = "to_be_replaced"

    PATH_TO_CLASSIFIER = "to_be_replaced"

    # Test on data that the flow has never seen before.
    PATH_TO_FLOW = "to_be_replaced"
    PATH_TO_FLOW_DENSITY_EST = "to_be_replaced"
    # Only used for the heuristic for getting multiple imputations using the flow
    DATA_PATH_TRAIN = "to_be_replaced"
    DATA_PATH_VAL = "to_be_replaced"
    DATA_PATH_TEST = "to_be_replaced"

    current_domain = "to_be_replaced"
    experiment_keys = ["concrete", "diabetes", "wine", "power"]
    experiment_paths = {
        experiment_keys[0]: "./experiment_sources/concrete",
        experiment_keys[1]: "./experiment_sources/diabetes",
        experiment_keys[2]: "./experiment_sources/wine",
        experiment_keys[3]: "./experiment_sources/power"
    }
    domain_dimensionality = {
        experiment_keys[0]: 9,
        experiment_keys[1]: 9,
        experiment_keys[2]: 12,
        experiment_keys[3]: 5
    }

    @classmethod
    def to_string(cls):
        config_str = ""
        for attribute, value in cls.__dict__.items():
            if not attribute.startswith('__') and not callable(value):
                config_str += f"{attribute}: {value}\n"
        return config_str
    @staticmethod
    def set_experiment_folder(domain_id, use_deep_classifier):
        Config.current_domain = domain_id
        Config.__use_deep_classifier = use_deep_classifier
        if domain_id not in Config.experiment_keys:
            print("wrong domain id")
            exit(-1)
        Config.classifier_threshold = Config.classifier_thold_factor / Config.domain_dimensionality[domain_id]

        Config.optimize_plausibility = Config.plausibility_constant_factor / Config.domain_dimensionality[domain_id]
        Config.FIO_opt_plausibility = Config.plausibility_constant_factor / Config.domain_dimensionality[domain_id]
        Config.DM_opt_plausibility = Config.plausibility_constant_factor / Config.domain_dimensionality[domain_id]

        Config.optimize_cost = 1 - Config.optimize_plausibility
        Config.FIO_opt_cost = 1 - Config.FIO_opt_plausibility
        Config.DM_opt_cost = 1 - Config.DM_opt_plausibility
        Config.FIO_opt_imputation = 3

        Config.__experiment_path = Config.experiment_paths[domain_id]
        Config.PATH_TO_FLOW = f'{Config.__experiment_path}/flow.onnx'
        Config.PATH_TO_FLOW_DENSITY_EST = f'{Config.__experiment_path}/backward_processed.onnx'
        if use_deep_classifier:
            Config.PATH_TO_CLASSIFIER = f'{Config.__experiment_path}/classifier_medium.onnx'
        else:
            Config.PATH_TO_CLASSIFIER = f'{Config.__experiment_path}/classifier_tiny.onnx'

        Config.DATA_PATH_TRAIN = f'{Config.__experiment_path}/train.csv'
        Config.DATA_PATH_VAL = f'{Config.__experiment_path}/val.csv'
        Config.DATA_PATH_TEST = f'{Config.__experiment_path}/test.csv'
