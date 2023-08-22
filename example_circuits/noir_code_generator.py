import json
import argparse


def generate_nn_code(model_data_file, test_samples_file, save_path):
    # Read the model_data
    with open(model_data_file, 'r') as f:
        model_data = json.load(f)

    idx = 1
    model_str = ""
    while f'l{idx}_weights' in model_data and f'l{idx}_biases' in model_data:
        model_str += f"global l{idx}_weights: [Field; {len(model_data[f'l{idx}_weights'])}] = {model_data[f'l{idx}_weights']};\n"
        model_str += f"global l{idx}_biases: [Field; {len(model_data[f'l{idx}_biases'])}] = {model_data[f'l{idx}_biases']};\n\n"
        idx += 1

    # Calculate the input dimension from l1_weights and l1_biases
    input_dim = len(model_data['l1_weights']) // len(model_data['l1_biases'])

    # Read the test_samples
    with open(test_samples_file, 'r') as f:
        test_samples = json.load(f)

    test_str = "////////////////////\n//     TESTS      //\n////////////////////\n"
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
        f.write(f"fn main(input: [Field; {input_dim}]) -> pub Field {{\n{main_logic}  output\n}}\n\n")
        f.write(test_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 'main.nr' from model data and test samples.")
    parser.add_argument("--model_data", type=str,
                        help="Path to the JSON file containing model data.")
    parser.add_argument("--test_samples", type=str,
                        help="Path to the JSON file containing test samples.")
    parser.add_argument("--save_path", type=str, default="main.nr",
                        help="Path to save the generated 'main.nr'.")

    args = parser.parse_args()

    generate_nn_code(args.model_data, args.test_samples, args.save_path)

    print(f"Generated Noir program at {args.save_path}")