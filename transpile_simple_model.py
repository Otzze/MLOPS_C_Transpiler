import joblib

def read_weights(file_path):
    read_model = joblib.load(file_path)
    return [read_model.intercept_] + list(read_model.coef_)

def load_fun_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def to_c_array(arr):
    res = "{"
    for i in range(len(arr)):
        res += f"{arr[i]}"
        if i < len(arr) - 1:
            res += ", "
    res += "}"
    return res

def generate_c_tree(tree, feature_names=None, node=0, depth=1):
    indent = "    " * depth
    if tree.children_left[node] == -1 and tree.children_right[node] == -1:
        value = tree.value[node].flatten()
        prediction = float(value.mean())
        return f"{indent}return {prediction};\n"

    feature = tree.feature[node]
    threshold = tree.threshold[node]
    feature_name = f"inputs[{feature}]" if feature_names is None else feature_names[feature]

    code = f"{indent}if ({feature_name} <= {threshold:.6f}) {{\n"
    code += generate_c_tree(tree, feature_names, tree.children_left[node], depth + 1)
    code += f"{indent}}} else {{\n"
    code += generate_c_tree(tree, feature_names, tree.children_right[node], depth + 1)
    code += f"{indent}}}\n"
    return code

def transpile_to_c(model_path, output_path, inputs, type="auto"):
    model = joblib.load(model_path)

    if type == "auto":
        if hasattr(model, "coef_"):
            type = "logistic" if hasattr(model, "classes_") else "linear"
        elif hasattr(model, "tree_"):
            type = "decision_tree"
        else:
            raise ValueError("Unsupported model type for auto-detection")

    res = "#include <stdio.h>\n\n"

    if type in ("linear", "logistic"):
        weights = read_weights(model_path)
        weight_array = to_c_array(weights)
        input_array = to_c_array(inputs)
        res += load_fun_from_file(
            "predict_function_logistic" if type == "logistic" else "predict_function"
        )
        res += "\n\nint main() {\n"
        res += f"    float weights[] = {weight_array};\n"
        res += f"    float inputs[] = {input_array};\n"
        res += f"    float result = "
        if type == "linear":
            res += f"linear_regression_prediction(inputs, weights, {len(inputs)});\n"
        else:
            res += f"logistic_regression(inputs, weights, {len(inputs)});\n"
        res += "    printf(\"Prediction: %f\\n\", result);\n"
        res += "    return 0;\n"
        res += "}\n"

    elif type == "decision_tree":
        tree = model.tree_
        tree_func = "float decision_tree_predict(float inputs[]) {\n"
        tree_func += generate_c_tree(tree)
        tree_func += "}\n\n"

        input_array = to_c_array(inputs)
        res += tree_func
        res += "int main() {\n"
        res += f"    float inputs[] = {input_array};\n"
        res += "    float result = decision_tree_predict(inputs);\n"
        res += "    printf(\"Prediction: %f\\n\", result);\n"
        res += "    return 0;\n"
        res += "}\n"

    else:
        raise ValueError(f"Unsupported model type: {type}")

    with open(output_path, 'w') as file:
        file.write(res)

OUTPUT_C_FILE = 'transpiled_model.c'
MODEL_FILE = '../decision_tree_model.joblib'
inputs = [12, 2, 0]

#model = joblib.load(MODEL_FILE)

#print(f"prediction should be {model.predict([inputs])}")

transpile_to_c(MODEL_FILE, OUTPUT_C_FILE, inputs)
