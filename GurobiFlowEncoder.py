import gurobipy
import numpy
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from typing import Literal
import csv
import math
from scipy.stats import norm
import Config


class GurobiFlowEncoder:
    def __init__(self, gurobi_model, onnx_path, naming_prefix):
        self.onnx_gurobi_map_vars = {}
        self.onnx_constants = {}
        self.big_M = 10000
        self.gurobi_model = gurobi_model
        self.onnx_path = onnx_path
        self.flow = onnx.load(self.onnx_path)
        self.input_node_name = self.flow.graph.input[0].name
        self.final_node_name = self.flow.graph.output[0].name
        # The name of the linear program is the same as the name of the onnx file.
        # e.g. /path/to/onnx_file.onnx -> onnx_file.lp
        self.linear_program_export_name = (f'{"/".join(onnx_path.split("/")[:-1])}/'
                                           f'{onnx_path.split("/")[-1].split(".")[0]}.lp')
        self.input_vars_gurobi = None
        self.output_vars_gurobi = None
        self.naming_prefix = naming_prefix

    def register_variables(self, key, value):
        self.onnx_gurobi_map_vars[key] = value
        if key == self.final_node_name:
            self.output_vars_gurobi = self.onnx_gurobi_map_vars[key]
            for k in range(len(self.onnx_gurobi_map_vars[key])):
                self.gurobi_model.addConstr(self.onnx_gurobi_map_vars[key][k] <= self.big_M)
                self.gurobi_model.addConstr(self.onnx_gurobi_map_vars[key][k] >= -self.big_M)

    def fetch_novel_variable_names(self, size, type: Literal["continuous", "binary", "integer"] = "continuous"):
        self.gurobi_model.update()
        variable_counter = len(self.gurobi_model.getVars())
        if type == "continuous":
            new_vars = [f'{self.naming_prefix}{i}' for i in range(variable_counter, variable_counter + size)]
            mul_result_vars = self.gurobi_model.addVars(new_vars, lb=-self.big_M, ub=self.big_M,
                                                        vtype=gurobipy.GRB.CONTINUOUS)
            subscribable_vars = [i for i in mul_result_vars.values()]
            return subscribable_vars
        elif type == "binary":
            new_vars = [f'{self.naming_prefix}{i}' for i in range(variable_counter, variable_counter + size)]
            binary_vars = self.gurobi_model.addVars(new_vars, vtype=gurobipy.GRB.BINARY)
            subscribable_vars = [i for i in binary_vars.values()]
            return subscribable_vars
        elif type == "integer":
            new_vars = [f'{self.naming_prefix}{i}' for i in range(variable_counter, variable_counter + size)]
            mul_result_vars = self.gurobi_model.addVars(new_vars, lb=-self.big_M, ub=self.big_M,
                                                        vtype=gurobipy.GRB.INTEGER)
            subscribable_vars = [i for i in mul_result_vars.values()]
            return subscribable_vars

    def fetch_constants(self):
        for init in self.flow.graph.initializer:
            self.onnx_constants[init.name] = numpy_helper.to_array(init)

        for node in self.flow.graph.node:
            node.doc_string = ""
            if node.op_type == "Constant":
                self.onnx_constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)

    def fetch_inputs(self, inputs_onnx):  # Convention: always return the variable input first.
        if (inputs_onnx[0] in self.onnx_gurobi_map_vars) and (inputs_onnx[1] in self.onnx_constants):
            return self.onnx_gurobi_map_vars[inputs_onnx[0]], self.onnx_constants[inputs_onnx[1]]
        elif (inputs_onnx[0] in self.onnx_constants) and (inputs_onnx[1] in self.onnx_gurobi_map_vars):
            return self.onnx_gurobi_map_vars[inputs_onnx[1]], self.onnx_constants[inputs_onnx[0]]
        elif (inputs_onnx[0] in self.onnx_gurobi_map_vars) and (inputs_onnx[1] in self.onnx_gurobi_map_vars):
            return self.onnx_gurobi_map_vars[inputs_onnx[0]], self.onnx_gurobi_map_vars[inputs_onnx[1]]

    def fetch_variable_handles(self, node, matmul_node=False):
        if not len(node.input) == 2:
            print(f'node {node} does not have exactly 2 inputs')
        if not len(node.output) == 1:
            print(f'node {node} does not have exactly 1 output')
        inputs_gurobi = self.fetch_inputs(node.input)
        # In our cases, the resulting variable has the same dimensionality as either of the input variables.
        if matmul_node:
            result_vars = self.fetch_novel_variable_names(inputs_gurobi[1].shape[1])
        else:
            result_vars = self.fetch_novel_variable_names(len(inputs_gurobi[0]))
        self.gurobi_model.update()
        return inputs_gurobi[0], inputs_gurobi[1], result_vars

    def process_mul_node(self, node):
        var_input, const_input, result_vars = self.fetch_variable_handles(node=node)
        for i in range(len(result_vars)):
            self.gurobi_model.addConstr(result_vars[i] == const_input[i] * var_input[i],
                                        name=f'{self.naming_prefix}|Mul|{node.name}|inputs_{i}')
        self.register_variables(node.output[0], result_vars)
        self.gurobi_model.update()

    def process_matmul_node(self, node):
        var_input, const_input, result_vars = self.fetch_variable_handles(node=node, matmul_node=True)
        for column in range(len(const_input[0])):
            self.gurobi_model.addConstr(result_vars[column] == gurobipy.quicksum(
                var_input[k] * const_input[k][column] for k in range(len(var_input))),
                                        name=f'{self.naming_prefix}|MatMul|{node.name}|column_{column}')
        self.register_variables(node.output[0], result_vars)
        self.gurobi_model.update()

    def process_add_node(self, node):
        var_input, const_input, result_vars = self.fetch_variable_handles(node=node)
        for row in range(len(var_input)):
            self.gurobi_model.addConstr(result_vars[row] == var_input[row] + const_input[row],
                                        name=f'{self.naming_prefix}|Add|{node.name}|Row_{row}')
        self.register_variables(node.output[0], result_vars)
        self.gurobi_model.update()

    def process_relu_node(self, node):
        if not len(node.input) == 1:
            print(f'node {node} does not have exactly 1 input')
        if not len(node.output) == 1:
            print(f'node {node} does not have exactly 1 output')
        # 1 if relu is active
        inputs_relu_var = self.onnx_gurobi_map_vars[node.input[0]]
        # inactive if ==1, this means output =0 ,
        # active relu otherwise, i.e., output of relu == input
        relu_active_indicator_var = self.fetch_novel_variable_names(len(inputs_relu_var), "binary")
        result_vars = self.fetch_novel_variable_names(len(inputs_relu_var))
        for i in range(len(result_vars)):
            # Explanation of big-m encoding with regular NNs (without interval weights)
            # a := relu_active_indicator_var
            # n := result_vars[i]
            # x := inputs_relu_var[i]

            # n >= 0             | eq 1
            # n <= M(1-a)        | eq 2
            # n <= x + Ma        | eq 3
            # n >= x             | eq 4

            # a=0 -> (f(x)=x)
            #    1. x > 0 -> n>=0, n<=M, n<=Wx+b , n>=Wx+b satisfiable n=Wx+b
            #    2. x < 0 -> n>=0, n<=M, n<=Wx+b , n>=Wx+b unsatisfiable
            # a=1 -> (f(x)=0)
            #    1. x > 0 -> n>=0, n<=0, n<=Wx+b +M, n>=Wx+b  unsatisfiable
            #    2. x < 0 -> n>=0, n<=0, n<=Wx+b +M, n>=Wx+b satisfiable n=0
            self.gurobi_model.addConstr(result_vars[i] >= 0, name=f'{self.naming_prefix}|Relu|{node.name}|EQ1|{i}')
            self.gurobi_model.addConstr(result_vars[i] <= self.big_M * (1 - relu_active_indicator_var[i]),
                                        name=f'{self.naming_prefix}|Relu|{node.name}|EQ2|{i}')
            self.gurobi_model.addConstr(
                result_vars[i] <= inputs_relu_var[i] + self.big_M * relu_active_indicator_var[i],
                name=f'{self.naming_prefix}|Relu|{node.name}|EQ3|{i}')
            self.gurobi_model.addConstr(result_vars[i] >= inputs_relu_var[i],
                                        name=f'{self.naming_prefix}|Relu|{node.name}|EQ4|{i}')
        self.register_variables(node.output[0], result_vars)
        self.gurobi_model.update()

    def process_div_node(self, node):
        var_input, const_input, result_vars = self.fetch_variable_handles(node=node)
        for row in range(len(var_input)):
            self.gurobi_model.addConstr(result_vars[row] == var_input[row] / const_input[row],
                                        name=f'{self.naming_prefix}|Div|{node.name}|Row_{row}')
        self.register_variables(node.output[0], result_vars)
        self.gurobi_model.update()

    def process_sub_node(self, node):
        var_input, const_input, result_vars = self.fetch_variable_handles(node=node)
        for row in range(len(var_input)):
            self.gurobi_model.addConstr(result_vars[row] == var_input[row] - const_input[row],
                                        name=f'{self.naming_prefix}|Sub|{node.name}|Row_{row}')
        self.register_variables(node.output[0], result_vars)
        self.gurobi_model.update()

    def encode(self):
        input_variables = self.fetch_novel_variable_names(
            self.flow.graph.input[0].type.tensor_type.shape.dim[0].dim_value)
        self.onnx_gurobi_map_vars[self.input_node_name] = input_variables
        self.input_vars_gurobi = input_variables
        # Load all the weights from the flow model, both in constant nodes and in initializers.
        self.fetch_constants()
        for node in self.flow.graph.node:
            if node.op_type == "Mul":
                self.process_mul_node(node)
            if node.op_type == "MatMul":
                self.process_matmul_node(node)
            if node.op_type == "Constant":
                # A constant node does not represent an operation. It merely saves a constant value.
                # Since that value is already read in our 'fetch_constants' method, we can ignore this node type.
                continue
            if node.op_type == "Add":
                self.process_add_node(node)
            if node.op_type == "Relu":
                self.process_relu_node(node)
            if node.op_type == "Div":
                self.process_div_node(node)
            if node.op_type == "Sub":
                self.process_sub_node(node)

    def compare_gurobi_onnx(self):
        input_node_values = [self.input_vars_gurobi[i].X for i in range(len(self.input_vars_gurobi))]
        print(f'input nodes: {input_node_values}')
        input_node_values_numpy = numpy.asarray(input_node_values, dtype=numpy.float32)

        output_node_values_solver = [self.output_vars_gurobi[i].X for i in range(len(self.output_vars_gurobi))]
        print(f'output nodes: {output_node_values_solver}')

        ort_sess = ort.InferenceSession(self.onnx_path)

        outputs = ort_sess.run(None, {self.flow.graph.input[0].name: input_node_values_numpy})
        onnx_result = [i for i in outputs[0]]
        print(f'output with onnxruntime: {onnx_result}')
        difference_found = 0
        for i in range(len(output_node_values_solver)):
            if not round(output_node_values_solver[i], 1) == round(float(onnx_result[i]), 1):
                difference_found = difference_found + 1
                print(f'{round(output_node_values_solver[i], 1)}\n {round(float(onnx_result[i]), 1)}')
        if not difference_found == 0:
            print(f'The variable assignments have errors.')
            exit(-1)
        else:
            print('onnx output matches with gurobi output!')

    def exhaustive_test(self):
        for test_val in numpy.arange(0.01, 1, 0.1):
            for i in range(len(self.input_vars_gurobi)):
                self.gurobi_model.addConstr(self.input_vars_gurobi[i] == test_val, name=f'tmp_constraint_{i}')
            self.gurobi_model.update()
            self.gurobi_model.optimize()
            self.compare_gurobi_onnx()
            for i in range(len(self.input_vars_gurobi)):
                self.gurobi_model.remove(self.gurobi_model.getConstrByName(f'tmp_constraint_{i}'))
            self.gurobi_model.update()

    def export_solver_model(self):
        self.gurobi_model.write(self.linear_program_export_name)


def plausible_ctx(classifier, target, flow_encoded, instance):
    # (1): c(ctx) == target, where target = c(ctx)
    # Restricting the ctx to integral results increases runtime by an order of magnitude.
    ctx = classifier.fetch_novel_variable_names(len(instance), "integer")
    for i in range(len(instance)):  # input and output dimensionality is same, so do both in one go!
        classifier.gurobi_model.addConstr(classifier.input_vars_gurobi[i] == ctx[i], name=f'ctx_to_classifier{i}')
    for i in range(len(classifier.output_vars_gurobi)):
        if target == 1:
            classifier.gurobi_model.addConstr(classifier.output_vars_gurobi[i] >= 0.501, name=f'ctx_1{i}')
        elif target == 0:
            classifier.gurobi_model.addConstr(classifier.output_vars_gurobi[i] <= 0.501, name=f'ctx_0{i}')
        else:
            print("Only binary classifier supported at the moment.")

    for i in range(len(instance)):  # input and output dimensionality is same, so do both in one go!
        flow_encoded.gurobi_model.addConstr(flow_encoded.input_vars_gurobi[i] == ctx[i], name=f'ctx_to_flow{i}')

    # Ensure that the ctx is plausible according to our flow model.
    thold_simplified_udl = 0.1
    quantile = quantile_log_normal(thold_simplified_udl)
    # Encode the abs and then build sum constraint over abs sum of probs
    aux_abs_var = classifier.fetch_novel_variable_names(len(instance))
    for i in range(len(instance)):
        flow_encoded.gurobi_model.addConstr(aux_abs_var[i] == gurobipy.abs_(flow_encoded.output_vars_gurobi[i]),
                                            name=f'ctx_abs_densities_pos{i}')
    flow_encoded.gurobi_model.addConstr(gurobipy.quicksum(aux_abs_var) <= quantile)

    aux_max_var = classifier.fetch_novel_variable_names(len(instance))
    for i in range(len(instance)):
        classifier.gurobi_model.addConstr(aux_max_var[i] >= ctx[i] - instance[i], name=f'abs_ctx_1_{i}')
        classifier.gurobi_model.addConstr(aux_max_var[i] >= instance[i] - ctx[i], name=f'abs_ctx_2_{i}')

    classifier.gurobi_model.setObjective(gurobipy.quicksum(aux_max_var), gurobipy.GRB.MINIMIZE)
    classifier.export_solver_model()
    classifier.gurobi_model.optimize()
    classifier_output = [classifier.input_vars_gurobi[i].X for i in range(len(classifier.input_vars_gurobi))]
    flow_output = [flow_encoded.output_vars_gurobi[i].X for i in range(len(flow_encoded.output_vars_gurobi))]
    return classifier_output, flow_output


def run_onnx(path_to_onnx, input):
    ort_sess = ort.InferenceSession(path_to_onnx)
    onnx_model = onnx.load(path_to_onnx)
    return ort_sess.run(None, {onnx_model.graph.input[0].name: numpy.asarray(input, dtype=numpy.float32)})


def quantile_log_normal(p, mu=1, sigma=0.5):
    return math.exp(mu + sigma * norm.ppf(p))


def invers_quantile_lognormal(thold):
    lo = 0
    hi = 1
    p = (hi - lo) / 2
    val = quantile_log_normal(p)
    while abs(val - thold) >= 0.0001:
        if val < thold:
            lo = lo + ((hi - lo) / 2)
        if val >= thold:
            hi = hi - ((hi - lo) / 2)
        p = lo + (hi - lo) / 2
        val = quantile_log_normal(p)
    return p, quantile_log_normal(p)


def main(identifier, instance_positive, ground_truth, TARGET_FILE):
    # ctx means counterfactual explanation
    clf_path = 'models/old_models_ignore/heloc_model.onnx'
    flow_bkw_path = 'models/old_models_ignore/flow_1_heloc_backward_21_inputs_without_reshape.onnx'

    single_model = gurobipy.Model('singleton')
    classifier_enc = GurobiFlowEncoder(gurobi_model=single_model,
                                       onnx_path=clf_path,
                                       naming_prefix="clf|")
    classifier_enc.encode()

    backward_encoder = GurobiFlowEncoder(gurobi_model=single_model,
                                         onnx_path=flow_bkw_path,
                                         naming_prefix="flw_bckw|")
    backward_encoder.encode()

    outputs = run_onnx(clf_path, instance_positive)
    prediction = int(outputs[0][0] > 0)
    if not prediction == ground_truth:
        print("ground truth does not match with classification. Skipping")
        return False
    target = 1 if outputs[0][0] < 0 else 0
    synthetic_ctx, flow_density_estimate = plausible_ctx(classifier_enc, target, backward_encoder, instance_positive)

    print(f'Classification of original instance: {outputs[0][0]}, opposite target class: {target}')
    ctx_class = run_onnx(clf_path, synthetic_ctx)
    print(f'target {target} and ctx is {ctx_class}')
    print(f'ctx: {synthetic_ctx}')
    print(f'input: {instance_positive}')
    with open(TARGET_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the list of floats as a single row
        writer.writerow([identifier] + instance_positive)
        writer.writerow([identifier] + synthetic_ctx)

    sum_flow_density = sum([abs(val) for val in flow_density_estimate])
    print(f'sum flow density: {sum_flow_density}')
    print(f'UDL: {invers_quantile_lognormal(sum_flow_density)}')
    density_estimate = run_onnx(flow_bkw_path, synthetic_ctx)
    print(f'density solver {flow_density_estimate}')
    print(f'density onnx {density_estimate[0]}')
    return True


if __name__ == '__main__':
    DATA_PATH = "./heloc/heloc_positive_original.csv"
    TARGET_FILE = "./heloc/ctx.csv"
    n = 1
    print(quantile_log_normal(0.1))
    successful_iterations = 0
    with open(DATA_PATH, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            if i == 0:  # skip header line
                continue
            if successful_iterations >= n:
                print(f'done all {n} counterfactuals')
                break
            print('line[{}] = {}'.format(i, line))
            float_row = [float(value) for value in line[0].split(",")[:-1]]
            ground_truth = int(round(float(line[0].split(",")[-1])))
            # The input instance needs to be rounded when restricting the ctx only to integral results.
            input_instance = [round(val) for val in float_row]
            print(input_instance)
            success = main(i, input_instance, ground_truth, TARGET_FILE)
            if success:
                successful_iterations = successful_iterations + 1
