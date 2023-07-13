use crate::is_greater_than_signed;

fn arg_max<N>(values: [ Field; N ]) -> Field {
  assert(N as u64 > 0);

  let mut max_index = 0;
  let mut max_value = values[0];

  for i in 1..N {
    let value = values[i];
    if is_greater_than_signed::is_greater_than_signed(value, max_value) {
      max_value = value;
      max_index = i;
    }
  }
  max_index
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_arg_max() {
    assert(arg_max([ 1 ]) == 0); 
    assert(arg_max([ 1, 2 ]) == 1); 
    assert(arg_max([ 2, 1]) == 0); 
    assert(arg_max([ 3, 2, 5, 1, 4 ]) == 2);
    assert(arg_max([ -1, 1]) == 1);
    assert(arg_max([ 1, -1]) == 0);
    assert(arg_max([ -3, -2, -5, -1, -4]) == 3);
}