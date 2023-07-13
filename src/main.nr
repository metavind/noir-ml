mod arg_max;
mod dense;
mod is_greater_than_signed;
mod is_greater_than;
mod is_negative;
mod is_positive;
mod mat_vec_mul;
mod relu;
mod models;

use dep::std;

fn main(input: pub [Field; 784], hid_weights: [Field; 15680], hid_bias: [Field; 20], out_weights: [Field; 200], out_bias: [Field; 10]) -> pub Field {
  let mut hid = dense::dense(input, hid_weights, hid_bias);
  hid = relu::relu(hid);
  let output = dense::dense(hid, out_weights, out_bias);
  let result = arg_max::arg_max(output);
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_main() {
  assert(1 == 1);
}

