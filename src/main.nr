mod activations;
mod utils;
mod layers;
mod models;

fn main(input: pub [Field; 784], hid_weights: [Field; 15680], hid_bias: [Field; 20], out_weights: [Field; 200], out_bias: [Field; 10]) -> pub Field {
  let mut hid = layers::dense(input, hid_weights, hid_bias);
  hid = activations::relu(hid);
  let output = layers::dense(hid, out_weights, out_bias);
  let result = utils::arg_max(output);
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_main() {
  assert(1 == 1);
}

