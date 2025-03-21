import gurobipy
from GurobiFlowEncoder import GurobiFlowEncoder
import math
from util import fetch_gurobi_env

TIMEOUT = "Timeout"
SUCCESS = "Success"


class FlowImputerSingle:
    def __init__(self):
        pass

    def flow_imputation(self, missing_instance, path_to_flow, model=None):
        if model is None:
            model = gurobipy.Model('imputation', env=fetch_gurobi_env())
        model.Params.LogToConsole = 0
        flow_encoded = GurobiFlowEncoder(gurobi_model=model,
                                         onnx_path=path_to_flow,
                                         naming_prefix="flow_impute_single|")
        flow_encoded.encode()

        # The flow model maps from latent space to target space.
        # Ensure that the ctx is plausible according to our flow model.
        # Encode the abs and then build sum constraint over abs sum of probs
        aux_abs_var = flow_encoded.fetch_novel_variable_names(len(missing_instance))
        for i in range(len(missing_instance)):
            flow_encoded.gurobi_model.addConstr(aux_abs_var[i] == gurobipy.abs_(flow_encoded.input_vars_gurobi[i]),
                                                name=f'density_aux_abs{i}')
        flow_encoded.gurobi_model.setObjective(gurobipy.quicksum(aux_abs_var), gurobipy.GRB.MINIMIZE)

        # Fix the output values for the known inputs. Only the missing values remain free.
        dimensions_dropped = []
        for i in range(len(missing_instance)):
            if not math.isnan(missing_instance[i]):
                flow_encoded.gurobi_model.addConstr(flow_encoded.output_vars_gurobi[i] == missing_instance[i],
                                                    name=f'fix_specified_vals{i}')
            else:
                flow_encoded.gurobi_model.addConstr(flow_encoded.output_vars_gurobi[i] >= 0,
                                                    name=f'fix_unspec_val_geq_0{i}')
                flow_encoded.gurobi_model.addConstr(flow_encoded.output_vars_gurobi[i] <= 1,
                                                    name=f'fix_unspec_val_leq_1{i}')
                dimensions_dropped.append(i)

        flow_encoded.gurobi_model.optimize()

        if (flow_encoded.gurobi_model.status == gurobipy.GRB.TIME_LIMIT
                or flow_encoded.gurobi_model.SolCount == 0):
            print("ERROR: Imputing failed! ")
            return [0.5 for dimension_to_drop in dimensions_dropped], [val if not math.isnan(val) else 0.5 for val in missing_instance], TIMEOUT
            # return [], [], TIMEOUT

        # Fetch and return only the imputed values.
        output_node_values_solver = [flow_encoded.output_vars_gurobi[i].X for i in
                                     range(len(flow_encoded.output_vars_gurobi))]
        # return both the full imputed instance and only the values that were imputed.
        return [output_node_values_solver[dimension_to_drop] for dimension_to_drop in
                dimensions_dropped], output_node_values_solver, SUCCESS


# Flow-imputer that can directly be encoded into the MILP.
# Unlike the one before, this is not a preprocessing-like imputation.

class FlowImputerInMILP:
    def __init__(self, gurobi_model, instance, missing_dimension, naming_prefix, path_to_flow):
        self.instance = instance  # The input vector. May contain any value (such as NaN) at the dimension to drop.
        self.missing_dimension = missing_dimension
        self.naming_prefix = naming_prefix
        self.flow_encoded = GurobiFlowEncoder(gurobi_model=gurobi_model,
                                              onnx_path=path_to_flow,
                                              naming_prefix=naming_prefix)

    def encode_imputing(self):
        self.flow_encoded.encode()

        # Ensure that the imputation is plausible according to our flow model.
        # Encode the abs and then build sum constraint over abs sum of probs
        aux_abs_var = self.flow_encoded.fetch_novel_variable_names(len(self.instance))
        for i in range(len(self.instance)):
            self.flow_encoded.gurobi_model.addConstr(
                aux_abs_var[i] == gurobipy.abs_(self.flow_encoded.input_vars_gurobi[i]),
                name=f'{self.naming_prefix}density_aux_abs{i}')
        for i in range(len(self.instance)):
            if not i in self.missing_dimension:
                self.flow_encoded.gurobi_model.addConstr(self.flow_encoded.output_vars_gurobi[i] == self.instance[i],
                                                         name=f'f{self.naming_prefix}ix_specified_vals{i}')
            else:
                self.flow_encoded.gurobi_model.addConstr(self.flow_encoded.output_vars_gurobi[i] >= 0,
                                                         name=f'f{self.naming_prefix}ix_specified_vals_geq_0{i}')
                self.flow_encoded.gurobi_model.addConstr(self.flow_encoded.output_vars_gurobi[i] <= 1,
                                                         name=f'f{self.naming_prefix}ix_specified_vals_leq_1{i}')

        # We also want to show the imputation result, so pass it back
        return aux_abs_var, self.flow_encoded.output_vars_gurobi  # return \ell and t


class FlowCtx:
    def __init__(self, gurobi_model, naming_prefix, path_to_flow):
        self.naming_prefix = naming_prefix
        self.flow_encoded = GurobiFlowEncoder(gurobi_model=gurobi_model,
                                              onnx_path=path_to_flow,
                                              naming_prefix=naming_prefix)

    def encode_plausible_counterfactual(self, target_class=None, target_index=None):
        # maybe encode the target class here, too, to incrase plausibility.
        self.flow_encoded.encode()
        # Ensure that the imputation is plausible according to our flow model.
        # Encode the abs and then build sum constraint over abs sum of probs
        no_input_nodes = len(self.flow_encoded.input_vars_gurobi)
        aux_abs_var = self.flow_encoded.fetch_novel_variable_names(no_input_nodes)
        if target_class is not None and target_index is not None:
            self.flow_encoded.gurobi_model.addConstr(
                self.flow_encoded.output_vars_gurobi[target_index]
                == target_class,
                name=f'{self.naming_prefix}_plausible_ctx_for_target_class'
            )
        else:
            print(f'ERROR, either target_class or target_index is missing')
            exit(-1)
        for i in range(no_input_nodes):
            self.flow_encoded.gurobi_model.addConstr(
                aux_abs_var[i] == gurobipy.abs_(self.flow_encoded.input_vars_gurobi[i]),
                name=f'{self.naming_prefix}density_aux_abs{i}')
        return aux_abs_var, self.flow_encoded.output_vars_gurobi  # return \ell' and t'


class Classifier:
    # the input vars contain the output of the flow model. They are set equal to the input of the classifier.
    def __init__(self, gurobi_model, naming_prefix, input_vars, target_class, path_to_classifier, classifier_threshold):
        self.input_vars = input_vars
        self.target_class = target_class
        self.naming_prefix = naming_prefix
        self.classifier_encoded = GurobiFlowEncoder(gurobi_model=gurobi_model,
                                                    onnx_path=path_to_classifier,
                                                    naming_prefix=naming_prefix)
        self.output_is_target_binary = None  # Yet uninitialized
        self.output_is_target_binary_final = None  # Yet uninitialized
        self.classifier_threshold = classifier_threshold

    def encode_classification_constraints(self):  # This is used with the flow ctx
        self.classifier_encoded.encode()
        no_input_nodes = len(self.classifier_encoded.input_vars_gurobi)
        if not len(self.input_vars) == no_input_nodes:
            print(f'AError: number of inputs classifier: {no_input_nodes} '
                  f'and length of flow output: {len(self.input_vars)}')
            exit(-1)

        for i in range(no_input_nodes):
            self.classifier_encoded.gurobi_model.addConstr(self.classifier_encoded.input_vars_gurobi[i]
                                                           == self.input_vars[i],
                                                           name=f'{self.naming_prefix}bounding_classifier_input_output{i}')
        no_output_nodes = len(self.classifier_encoded.output_vars_gurobi)
        for i in range(no_output_nodes):
            if not i == self.target_class:
                self.classifier_encoded.gurobi_model.addConstr(
                    self.classifier_encoded.output_vars_gurobi[i]
                    <= (self.classifier_encoded.output_vars_gurobi[self.target_class] - self.classifier_threshold),
                    name=f'{self.naming_prefix}class_is_target{i}')

    def encode_target_classification_constraints(self, diff_vars):  # This is used with ARMIN
        self.classifier_encoded.encode()
        no_input_nodes = len(self.classifier_encoded.input_vars_gurobi)
        if not len(self.input_vars) == no_input_nodes:
            print(f'BError: number of inputs classifier: {no_input_nodes} '
                  f'and length of flow output: {len(self.input_vars)}')
            exit(-1)

        for not_target in range(no_input_nodes):
            self.classifier_encoded.gurobi_model.addConstr(self.classifier_encoded.input_vars_gurobi[not_target]
                                                           == self.input_vars[not_target] + diff_vars[not_target],
                                                           name=f'{self.naming_prefix}bounding_classifier_input_output{not_target}')

        self.output_is_target_binary_final = self.classifier_encoded.fetch_novel_variable_names(1, type="binary")[0]
        # output_is_target_binary is an indicator variable getting the value 1
        # if the classification is the target class. otherwise 0
        not_target = 1 - self.target_class

        self.classifier_encoded.gurobi_model.addConstr((self.output_is_target_binary_final == 1)
                                                       >>
                                                       (self.classifier_encoded.output_vars_gurobi[not_target]
                                                        <=
                                                        self.classifier_encoded.output_vars_gurobi[
                                                            self.target_class]),
                                                       name=f'{self.naming_prefix}classification_result_pos_{not_target}')

        self.classifier_encoded.gurobi_model.addConstr((self.output_is_target_binary_final == 0)
                                                       >>
                                                       (self.classifier_encoded.output_vars_gurobi[not_target]
                                                        >=
                                                        self.classifier_encoded.output_vars_gurobi[
                                                            self.target_class]),
                                                       name=f'{self.naming_prefix}classification_result_neg_{not_target}')

    def encode_single_imput_ctx_classification_constraints(self, diff_vars,
                                                           confidence_thold):  # This is used with ARMIN
        self.classifier_encoded.encode()
        no_input_nodes = len(self.classifier_encoded.input_vars_gurobi)
        if not len(self.input_vars) == no_input_nodes:
            print(f'BError: number of inputs classifier: {no_input_nodes} '
                  f'and length of flow output: {len(self.input_vars)}')
            exit(-1)

        for not_target in range(no_input_nodes):
            self.classifier_encoded.gurobi_model.addConstr(self.classifier_encoded.input_vars_gurobi[not_target]
                                                           == self.input_vars[not_target] + diff_vars[not_target],
                                                           name=f'{self.naming_prefix}bounding_classifier_input_output{not_target}')

        # output_is_target_binary is an indicator variable getting the value 1
        # if the classification is the target class. otherwise 0
        not_target = 1 - self.target_class

        self.classifier_encoded.gurobi_model.addConstr(
            ((self.classifier_encoded.output_vars_gurobi[not_target] + confidence_thold)
             <=
             self.classifier_encoded.output_vars_gurobi[self.target_class]),
            name=f'{self.naming_prefix}classification_result_pos_{not_target}')
