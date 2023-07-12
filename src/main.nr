mod arg_max;
mod dense;
mod is_greater_than_signed;
mod is_greater_than;
mod is_negative;
mod is_positive;
mod mat_vec_mul;
mod relu;

use dep::std;

fn main(input: pub [Field; 2], hid_weights: [Field; 20], hid_bias: [Field; 10], out_weights: [Field; 20], out_bias: [Field; 2]) -> pub Field {
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
fn test_test() {
  assert(1 == 1);
  let input: [Field; 2] = [321429, -305944];

  let hid_weights: [Field; 20] = [-201799, 253330, -94632, -1679517, -194549, 139218, -158168, 497705, -566381, 1437657, -1098219, -252057, -643523, -291986, -291486, 26194, 1668978, 576082, -1310992, -1006468];
  let hid_bias: [Field; 10] = [332004696131, -615841984749, -247557893395, 528358042240, -533230543137, -386732697487, 680972695351, -655961990356, -657531619072, -634221017361];

  let out_weights: [Field; 20] = [-167334, 1135549, -176126, -262836, 915547, 590526, -63137, 271829, 1246469, 1245495, 60699, -1024452, 24655, -177994, -1364973, -1049718, 78062, 185411, -1245607, -1103638];
  let out_bias: [Field; 2] = [-94743110239505760, 611362576484680192];

  let result: Field = main(input, hid_weights, hid_bias, out_weights, out_bias);

  std::println(result);
}

