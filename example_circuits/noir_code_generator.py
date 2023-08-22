import json
import argparse


def generate_nn_code(save_path, model_parameters_file, test_samples_file=None):
    # Read the model_parameters
    with open(model_parameters_file, 'r') as f:
        model_parameters = json.load(f)

    idx = 1
    model_str = ""
    while f'l{idx}_weights' in model_parameters and f'l{idx}_biases' in model_parameters:
        model_str += f"global l{idx}_weights: [Field; {len(model_parameters[f'l{idx}_weights'])}] = {model_parameters[f'l{idx}_weights']};\n"
        model_str += f"global l{idx}_biases: [Field; {len(model_parameters[f'l{idx}_biases'])}] = {model_parameters[f'l{idx}_biases']};\n\n"
        idx += 1

    # Calculate the input dimension from l1_weights and l1_biases
    input_dim = len(model_parameters['l1_weights']
                    ) // len(model_parameters['l1_biases'])

    # Read the test_samples
    if test_samples_file is not None:
        with open(test_samples_file, 'r') as f:
            test_samples = json.load(f)

        test_str = "\n////////////////////\n//     TESTS      //\n////////////////////\n"
        for i in range(1, len(test_samples) // 2 + 1):
            test_str += f"#[test]\nfn test_main_{i:03}() {{\n  let sample = {test_samples[f'input{i}']};\n  assert(main(sample) == {test_samples[f'output{i}']});\n}}\n\n"

    # Building the main function logic based on the number of layers
    main_logic = "  let output = input;\n"  # initialize
    for i in range(1, idx):
        if i != idx - 1:  # if it's not the last layer
            main_logic += f"  let output = relu(dense(output, l{i}_weights, l{i}_biases));\n"
        else:  # if it's the last layer
            main_logic += f"  let output = arg_max(dense(output, l{i}_weights, l{i}_biases));\n"

    # Write the content to main.nr
    with open(save_path, 'w') as f:
        f.write(
            "use dep::noir_ml::{layers::dense, activations::relu, utils::arg_max};\n\n")
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

    print(f"Generated Noir program at {args.save_path}")
