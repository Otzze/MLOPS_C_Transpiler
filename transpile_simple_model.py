import joblib

def read_weights(file_path):
    return joblib.load(file_path).coef_

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

def transpile_to_c(model_path, output_path, inputs):
    weights = read_weights(model_path)
    weight_array = to_c_array(weights)
    input_array = to_c_array(inputs)
    res = "#include <stdio.h>\n\n"
    res += "#include <stddef.h>\n\n"
    res += load_fun_from_file('predict_function')
    res += "\n\n"
    res += "int main() {\n"
    res += f"    float weights[] = {weight_array};\n"
    res += f"    float inputs[] = {input_array};\n"
    res += f"    float result = linear_regression_prediction(inputs, weights, {len(inputs)});\n"
    res += "    printf(\"Prediction: %f\\n\", result);\n"
    res += "    return 0;\n"
    res += "}\n"
    with open(output_path, 'w') as file:
        file.write(res)

OUTPUT_C_FILE = 'transpiled_model.c'
MODEL_FILE = 'regression.joblib'
inputs = [12, 2, 0]

model = joblib.load(MODEL_FILE)

print(f"prediction should be {model.predict([inputs])}")

transpile_to_c(MODEL_FILE, OUTPUT_C_FILE, inputs)
