use crate::utils;

fn relu<N>(values : [ Field; N ]) -> [ Field; N ] {
  let mut result = [0; N];

  for i in 0..N {
    if utils::is_positive(values[i]) {
      result[i] = values[i];
    }
  }
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_relu() {
  assert(relu([ -1, 0, 1]) == [ 0, 0, 1 ]);
  let comp_constant = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
  assert(relu([ comp_constant - 1, comp_constant, comp_constant + 1 ]) == [ comp_constant - 1, comp_constant, 0]);
}