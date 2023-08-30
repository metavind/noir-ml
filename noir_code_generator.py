"""
Neural Network Code Generator for Noir

Description:
    This script provides functionality to automatically generate Aleo code
    for representing and computing neural network models. The resulting Noir code
    contains data structures for the neural network's weights and biases along with
    the requisite logic for carrying out forward pass through the network.

Inputs:
    - Path to save the generated 'main.nr'.
    - A JSON file containing the neural network's weights and biases. The JSON should
      have keys of the form 'wX' and 'bX', where X indicates the layer number
      (starting from 1). The JSON file should be structured as:
        {
            "w1": [layer_1_weights],
            "b1": [layer_1_biases],
            "w2": [layer_2_weights],
            "b2": [layer_2_biases],
            ...
        }
    - (Optional) A JSON file containing test samples, structured with 'inputX' and 'outputX' keys,
      where X represents the sample number.

Outputs:
    - A Noir source code file ('main.nr') that illustrates the given neural network.
      This includes a `main` function that receives input data and yields an output.
    - If test samples are provided, the script will also generate test functions within
      the Noir code to verify the network's computations against the samples.

Usage:
    Run the script, supplying the necessary command-line arguments:
    --save_path: Destination path to save the generated 'main.nr'.
    --model_parameters: Path to the JSON file detailing the model parameters.
    --test_samples: (Optional) Path to the JSON file containing the test samples.

Example:
    python noir_program_generator.py --save_path src/main.nr --model_parameters model_parameters.json --test_samples test_samples.json

Code Generation Process:
    1. The script begins by parsing the JSON input to deduce the architecture of
       the neural network, taking into account the number of layers and their dimensions.
    2. Corresponding global variables are defined in the Noir code for each layer's
       weights and biases.
    3. The `main` function in Noir is crafted to execute the forward pass using the
       weights and biases set in the global variables.
    4. ReLU activation function is employed for all layers excluding the output layer.
    5. Finally, the Noir code computes the `arg_max` of the output layer to get the
       index with the highest value, denoting the network's prediction.

Note:
    The script leverages the 'noir_ml' library within Noir to utilize functions like `fc`,
    `relu`, and `arg_max` for the neural network computations.
"""

import json
import argparse


def generate_nn_code(save_path, model_parameters_file, test_samples_file=None):
    # Read the model_parameters
    with open(model_parameters_file, 'r') as f:
        model_parameters = json.load(f)

    idx = 1
    model_str = ""
    while f'w{idx}' in model_parameters and f'b{idx}' in model_parameters:
        model_str += f"global w{idx}: [Field; {len(model_parameters[f'w{idx}'])}] = {model_parameters[f'w{idx}']};\n"
        model_str += f"global b{idx}: [Field; {len(model_parameters[f'b{idx}'])}] = {model_parameters[f'b{idx}']};\n\n"
        idx += 1

    # Calculate the input dimension from w1 and b1
    input_dim = len(model_parameters['w1']
                    ) // len(model_parameters['b1'])

    # Read the test_samples
    if test_samples_file is not None:
        with open(test_samples_file, 'r') as f:
            test_samples = json.load(f)

        test_str = "\n////////////////////\n//     TESTS      //\n////////////////////\n"
        for i in range(1, len(test_samples) // 2 + 1):
            test_str += f"#[test]\nfn test_main_{i:03}() {{\n  let sample = {test_samples[f'in{i}']};\n  assert(main(sample) == {test_samples[f'out{i}']});\n}}\n\n"

    # Building the main function logic based on the number of layers
    main_logic = "  let output = input;\n"  # initialize
    for i in range(1, idx):
        if i != idx - 1:  # if it's not the last layer
            main_logic += f"  let output = relu(fc(output, w{i}, b{i}));\n"
        else:  # if it's the last layer
            main_logic += f"  let output = arg_max(fc(output, w{i}, b{i}));\n"

    # Write the content to main.nr
    with open(save_path, 'w') as f:
        f.write(
            "use dep::noir_ml::{layers::fc, activations::relu, utils::arg_max};\n\n")
        f.write(model_str)
        f.write(
            f"fn main(input: [Field; {input_dim}]) -> pub Field {{\n{main_logic}  output\n}}\n")
        if test_samples_file is not None:
            f.write(test_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 'main.nr' at desired path from model parameters and test samples.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the generated 'main.nr'.")
    parser.add_argument("--model_parameters", type=str, required=True,
                        help="Path to the JSON file containing model parameters.")
    parser.add_argument("--test_samples", type=str,
                        help="Path to the JSON file containing test samples.")

    args = parser.parse_args()

    generate_nn_code(args.save_path, args.model_parameters, args.test_samples)

    print(f"Generated Noir program: {args.save_path}")
